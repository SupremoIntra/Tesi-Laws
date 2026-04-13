"""
Adversarial Patch Optimizer con Expectation over Transformation (EoT).

Riferimento principale:
    Athalye, A., Sutskever, I. (2017). "Synthesizing Robust Adversarial Examples".
    https://arxiv.org/abs/1707.07397

Perché funziona (concetto chiave per la tesi):
    YOLO usa due passaggi separati:
      1. Forward grezzo (differenziabile): model.model(batch) → logit grezzi
      2. Post-processing NMS (NON differenziabile): filtra e seleziona box

    Noi ottimizziamo sul passaggio 1, che mantiene il computation graph intatto.
    Il gradiente fluisce: patch → trasformazioni EoT → immagine patchata → YOLO raw → loss

Struttura del file:
    get_chest_bbox()   — calcola dove posizionare la patch sul petto
    PatchOptimizer     — classe principale con ottimizzazione EoT
        optimize()     — metodo principale: ottimizza la patch su un'immagine
        _raw_loss()    — estrae la loss differenziabile dall'output grezzo di YOLO ← FIX CHIAVE
        apply_patch_to_image() — tensor slicing: sovrascrive i pixel del bbox
        _random_transform()    — trasformazioni EoT per robustezza fisica
"""

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from config import (
    PATCH_H, PATCH_W, PATCH_LR, PATCH_STEPS,
    EOT_N_TRANSFORMS, PERSON_CLASS_ID, IMG_SIZE
)

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ══════════════════════════════════════════════════════════════════════
# UTILITY: POSIZIONAMENTO PATCH
# ══════════════════════════════════════════════════════════════════════

def get_chest_bbox(person_bbox: Tuple[int,int,int,int],
                   patch_w: int, patch_h: int,
                   img_size: int = 640) -> Tuple[int,int,int,int]:
    """
    Calcola il bbox dove posizionare la patch, centrato sul petto.

    Logica:
        - Centro orizzontale (cx) = metà del bbox persona
        - Centro verticale (cy)   = 33% dall'alto del bbox
          (zona petto, non testa né pancia)
        - La patch ha dimensione fissa patch_w × patch_h

    Args:
        person_bbox: (x1,y1,x2,y2) del rilevamento YOLO
        patch_w, patch_h: dimensioni della patch in pixel
        img_size: dimensione dell'immagine (default 640)

    Returns:
        (px1,py1,px2,py2): bbox della patch, clampato ai bordi immagine
    """
    x1, y1, x2, y2 = person_bbox
    cx = (x1 + x2) // 2
    cy = y1 + int((y2 - y1) * 0.33)   # 33% dall'alto = zona petto
    px1 = max(0,        cx - patch_w // 2)
    py1 = max(0,        cy - patch_h // 2)
    px2 = min(img_size, cx + patch_w // 2)
    py2 = min(img_size, cy + patch_h // 2)
    return (px1, py1, px2, py2)


# ══════════════════════════════════════════════════════════════════════
# PATCH OPTIMIZER
# ══════════════════════════════════════════════════════════════════════

class PatchOptimizer:
    """
    Genera una adversarial patch ottimizzata tramite gradient descent + EoT.

    Flusso di ottimizzazione (per ogni step):
        1. Applica N trasformazioni fisiche casuali alla patch  ← EoT
        2. Incolla ogni patch trasformata sull'immagine        ← tensor slicing
        3. Esegui il forward GREZZO di YOLO (prima di NMS)     ← differenziabile
        4. Calcola loss = probabilità media classe "person"    ← vogliamo minimizzare
        5. loss.backward() → aggiorna la patch con Adam

    Perché il forward GREZZO e non l'API standard:
        L'API model(img_pil) include NMS e post-processing che rompono
        il computation graph di PyTorch. Il forward grezzo model.model(batch)
        restituisce i logit direttamente dalla rete, che sono differenziabili.
    """

    def __init__(self, patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 model_path: str = "yolov8n.pt"):
        if not HAS_YOLO:
            raise RuntimeError(
                "Ultralytics non installata. Esegui: pip install ultralytics")
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.model_path = model_path
        self._yolo      = None

        # La patch è il parametro da ottimizzare.
        # Inizializzata con rumore casuale uniforme in [0,1].
        # requires_grad=True: PyTorch traccia tutte le operazioni su questo tensore
        # per poter calcolare il gradiente durante backward().
        self.patch = torch.rand(3, patch_h, patch_w,
                                dtype=torch.float32,
                                requires_grad=True)

    # ── Caricamento modello ───────────────────────────────────────────
    def _get_model(self) -> "_YOLO":
        """Carica YOLOv8 la prima volta (lazy loading)."""
        if self._yolo is None:
            self._yolo = _YOLO(self.model_path)
            # Congela TUTTI i parametri di YOLO.
            # Noi ottimizziamo SOLO self.patch, non i pesi della rete.
            for p in self._yolo.model.parameters():
                p.requires_grad_(False)
            self._yolo.model.eval()
        return self._yolo

    # ── Loss differenziabile (FIX CHIAVE) ────────────────────────────
    @staticmethod
    def _raw_loss(torch_model: torch.nn.Module,
                  batch: torch.Tensor,
                  person_class: int = PERSON_CLASS_ID) -> torch.Tensor:
        """
        Esegue il forward GREZZO di YOLOv8 ed estrae una loss differenziabile.

        PERCHÉ QUESTO METODO RISOLVE IL BUG:
        ─────────────────────────────────────
        Il codice precedente faceva:
            preds = torch_model(batch)
            for pred in preds:      ← iterava sulla TUPLA (pred_tensor, features)
                if pred.dim() == 4: ← non matchava mai → count=0
            loss = torch.tensor(0.0, requires_grad=True)  ← foglia isolata!
            loss.backward()  ← CRASH: nessun grad_fn collegato alla patch

        La fix:
            raw = torch_model(batch)
            pred = raw[0]            ← estrae SOLO il tensore di predizione
            # pred shape: [1, 84, 8400]
            # 84 = 4 coordinate bbox + 80 classi COCO
            # 8400 = griglia 80×80 + 40×40 + 20×20 (multi-scale anchors)
            person_logits = pred[:, 4 + person_class, :]  # [1, 8400]
            # Il gradiente fluisce: patch → batch → raw → pred → loss ✓

        Args:
            torch_model: model.model (rete PyTorch interna di YOLO)
            batch: tensore immagine [1, 3, 640, 640] con patch applicata
            person_class: indice classe (0 = "person" in COCO)

        Returns:
            loss: scalare differenziabile = probabilità media "person"
                  Minimizzare questo valore → YOLO non vede più persone
        """
        raw = torch_model(batch)

        # YOLOv8 in eval mode restituisce una tupla: (pred_combined, feature_list)
        # pred_combined ha shape [batch, 84, 8400]
        # Se per qualche versione restituisce direttamente il tensore, gestiamo entrambi.
        if isinstance(raw, (tuple, list)):
            pred = raw[0]   # [1, 84, 8400]
        else:
            pred = raw      # [1, 84, 8400]

        # Struttura del tensore di output YOLOv8:
        #   pred[:, 0:4,  :]  → coordinate bbox (cx, cy, w, h) — non ci interessa
        #   pred[:, 4:84, :]  → logit per le 80 classi COCO
        #   pred[:, 4+0,  :]  → logit classe "person" (classe 0)
        person_logits = pred[:, 4 + person_class, :]   # [1, 8400]

        # Sigmoid converte i logit in probabilità [0,1]
        # YOLOv8 usa sigmoid (non softmax) per le classi — sono binary classifiers
        person_prob = torch.sigmoid(person_logits)      # [1, 8400]

        # Loss = media delle probabilità su tutti gli anchor
        # Minimizzare questo → la rete assegna bassa probabilità "person" ovunque
        return person_prob.mean()

    # ── EoT: trasformazione fisica casuale ────────────────────────────
    @staticmethod
    def _random_transform(patch: torch.Tensor) -> torch.Tensor:
        """
        Applica una trasformazione fisica casuale alla patch (EoT).

        Ogni trasformazione simula una condizione fisica reale:
            T1: rotazione ±20°      → drone non perfettamente allineato
            T2: scala 75%–125%      → variazione di distanza/altitudine
            T3: brightness/contrast → condizioni di illuminazione
            T4: flip orizzontale    → orientamento del soggetto
            T5: blur gaussiano      → defocus, movimento, bassa risoluzione

        Tutte queste operazioni sono DIFFERENZIABILI → il gradiente
        fluisce correttamente dalla loss fino alla patch originale.

        Note: patch.unsqueeze(0) aggiunge la dimensione batch richiesta
        da torchvision (che lavora su [B,C,H,W]). squeeze(0) la rimuove.
        """
        t = patch.unsqueeze(0)   # [1, 3, H, W]

        # T1: rotazione
        angle = random.uniform(-20, 20)
        t = TF.rotate(t, angle)

        # T2: scala (ridimensionamento + ripristino dimensioni originali)
        scale = random.uniform(0.75, 1.25)
        new_h = max(1, int(patch.shape[1] * scale))
        new_w = max(1, int(patch.shape[2] * scale))
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        t = F.interpolate(t, size=(patch.shape[1], patch.shape[2]),
                          mode="bilinear", align_corners=False)

        # T3: variazioni fotometriche
        t = TF.adjust_brightness(t, random.uniform(0.7, 1.3))
        t = TF.adjust_contrast(t,   random.uniform(0.8, 1.2))

        # T4: flip orizzontale (50% probabilità)
        if random.random() < 0.5:
            t = TF.hflip(t)

        # T5: blur gaussiano (simula sfocatura fisica)
        sigma = random.uniform(0, 1.5)
        if sigma > 0.3:
            k = max(3, int(sigma * 4) | 1)   # kernel dispari ≥ 3
            t = TF.gaussian_blur(t, kernel_size=k, sigma=sigma)

        return torch.clamp(t.squeeze(0), 0, 1)   # [3, H, W]

    # ── Tensor slicing: applica patch all'immagine ───────────────────
    @staticmethod
    def apply_patch_to_image(img_t: torch.Tensor,
                             patch_t: torch.Tensor,
                             bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Sovrascrive i pixel del bbox nell'immagine con la patch.

        Questa è l'operazione fondamentale dell'attacco adversariale fisico:
        nel mondo digitale sovrascriviamo i pixel con torch slicing;
        nel mondo fisico stampiamo la patch e la indossiamo → stesso effetto.

        img_clone[:, y1:y2, x1:x2] = patch_resized
                 ↑                    ↑
              tutti i canali RGB   patch ridimensionata al bbox

        Il .clone() è necessario perché PyTorch non permette in-place
        assignment su tensori nel computation graph senza esplicitare
        la copia → garantisce che il gradiente fluisca correttamente.
        """
        x1, y1, x2, y2 = bbox
        ph = y2 - y1
        pw = x2 - x1
        if ph <= 0 or pw <= 0:
            return img_t

        patch_resized = F.interpolate(
            patch_t.unsqueeze(0), size=(ph, pw),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        img_clone = img_t.clone()
        img_clone[:, y1:y2, x1:x2] = patch_resized
        return img_clone

    # ── Estrazione confidence per valutazione (NO gradiente) ─────────
    @staticmethod
    def _get_person_conf(results) -> float:
        """
        Estrae la confidence massima sulla classe 'person' dai risultati YOLO.
        Usata SOLO per valutazione (before/after), NON nell'ottimizzazione.
        Non differenziabile — usa l'API standard con NMS.
        """
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0

    @staticmethod
    def _find_person_bbox(results) -> Optional[Tuple[int, int, int, int]]:
        """Trova il bbox della prima persona rilevata da YOLO."""
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    box = r.boxes.xyxy[mask][0].int().tolist()
                    return tuple(box)
        return None

    # ── Metodo principale: ottimizzazione ────────────────────────────
    def optimize(self, image_path: str,
                 bbox: Optional[Tuple[int, int, int, int]] = None,
                 n_steps: int = PATCH_STEPS,
                 lr: float    = PATCH_LR,
                 n_eot: int   = EOT_N_TRANSFORMS,
                 verbose: bool = True) -> dict:
        """
        Ottimizza la patch su un'immagine con EoT.

        Flusso completo:
            1. Carica immagine (file o webcam)
            2. Rilevamento baseline con YOLO (API standard, NO grad)
            3. Trova bbox persona → calcola bbox petto
            4. Loop ottimizzazione:
                a. Per ogni step: N trasformazioni EoT
                b. Ogni trasformazione: applica patch → forward grezzo → loss
                c. Media loss su N trasformazioni → backward → Adam step
            5. Valutazione finale con YOLO (API standard)
            6. Restituisce patch, metriche, immagini per il plot

        Args:
            image_path: percorso file immagine oppure "webcam"
            bbox:       bbox manuale (x1,y1,x2,y2); se None, usa YOLO
            n_steps:    iterazioni di ottimizzazione
            lr:         learning rate Adam
            n_eot:      trasformazioni EoT per step
            verbose:    stampa loss ogni 10 step
        """
        model = self._get_model()

        # ── Step 1: carica immagine ───────────────────────────────────
        if image_path == "webcam":
            if not HAS_CV2:
                raise RuntimeError("pip install opencv-python")
            # CAP_AVFOUNDATION su macOS, fallback generico
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            for _ in range(10):    # scarta i primi frame (bilanciamento AE)
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError("Webcam non disponibile")
            cv2.imwrite("webcam_capture.jpg", frame)
            print("Frame webcam salvato → webcam_capture.jpg")
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.open(image_path).convert("RGB")

        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_np  = np.array(img_pil).astype(np.float32) / 255.0
        img_t   = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, 640, 640]

        # ── Step 2: baseline (API standard, non serve grad) ──────────
        print("Rilevamento baseline (senza patch)...")
        results_base = model(img_pil, verbose=False)
        conf_before  = self._get_person_conf(results_base)
        print(f"  Confidence PRIMA patch: {conf_before:.4f}")

        if conf_before < 0.01:
            print("  ATTENZIONE: nessuna persona rilevata nel baseline.")
            print("  Verifica che nell'immagine ci sia una persona visibile.")

        # ── Step 3: calcola bbox petto ────────────────────────────────
        person_bbox = bbox if bbox is not None else self._find_person_bbox(results_base)
        if person_bbox is None:
            # Fallback: centro immagine
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            person_bbox = (cx - 50, cy - 80, cx + 50, cy + 80)
            print("  Nessun bbox persona trovato → uso posizione centrale default")

        patch_bbox = get_chest_bbox(person_bbox, self.patch_w, self.patch_h, IMG_SIZE)
        x1, y1, x2, y2 = patch_bbox
        bbox_area  = max((x2-x1)*(y2-y1), 1)
        patch_area = self.patch_h * self.patch_w
        coverage   = patch_area / bbox_area

        print(f"  BBox persona: {person_bbox}")
        print(f"  BBox patch (petto): {patch_bbox}  coverage={coverage:.3f} ({coverage*100:.1f}%)")

        # ── Step 4: ottimizzazione EoT ────────────────────────────────
        # torch_model = model.model: rete PyTorch interna, PRIMA di NMS
        torch_model = model.model
        device      = next(torch_model.parameters()).device
        img_t       = img_t.to(device)

        optimizer    = torch.optim.Adam([self.patch], lr=lr)
        loss_history = []

        if verbose:
            print(f"\nOttimizzazione EoT: {n_steps} step, {n_eot} trasf./step")

        for step in range(n_steps):
            optimizer.zero_grad()
            step_losses = []

            for _ in range(n_eot):
                # Trasformazione casuale della patch (simula fisica)
                patch_t = self._random_transform(self.patch).to(device)

                # Applica patch sull'immagine (tensor slicing)
                img_patched = self.apply_patch_to_image(img_t, patch_t, patch_bbox)
                batch = img_patched.unsqueeze(0)  # [1, 3, 640, 640]

                # Forward grezzo differenziabile + estrazione loss
                # NOTA: usa _raw_loss(), NON model(batch) che rompe il graph
                loss = self._raw_loss(torch_model, batch)
                step_losses.append(loss)

            # Media loss su tutte le trasformazioni EoT
            final_loss = torch.stack(step_losses).mean()
            final_loss.backward()   # propaga il gradiente fino a self.patch
            optimizer.step()        # aggiorna self.patch

            # Mantieni i pixel nel range valido [0,1]
            with torch.no_grad():
                self.patch.clamp_(0, 1)

            loss_val = final_loss.item()
            loss_history.append(loss_val)

            if verbose and (step % 10 == 0 or step == n_steps - 1):
                print(f"  Step {step+1:3d}/{n_steps}  loss={loss_val:.4f}")

        # ── Step 5: valutazione finale (API standard) ─────────────────
        print("\nValutazione con patch ottimizzata...")
        img_final     = self.apply_patch_to_image(img_t, self.patch.detach(), patch_bbox)
        img_final_pil = Image.fromarray(
            (img_final.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        results_after = model(img_final_pil, verbose=False)
        conf_after    = self._get_person_conf(results_after)

        drop = conf_before - conf_after
        print(f"  Confidence DOPO patch: {conf_after:.4f}")
        print(f"  Drop: {drop:+.4f}  ({drop/max(conf_before,1e-6)*100:.1f}%)")

        return {
            "patch"          : self.patch.detach().cpu(),
            "conf_before"    : conf_before,
            "conf_after"     : conf_after,
            "conf_drop"      : drop,
            "loss_history"   : loss_history,
            "patch_coverage" : coverage,
            "bbox"           : patch_bbox,
            "img_original"   : img_t.cpu(),
            "img_patched"    : img_final.cpu(),
        }