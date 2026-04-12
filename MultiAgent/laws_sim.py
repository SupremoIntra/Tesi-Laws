#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║                       LAWS-SIM                                        ║
║         Multi-Agent LAWS Simulator + Real YOLOv8 + EoT Patch          ║
║                                                                       ║                                                                       ║
║  Uso:                                                                 ║
║    python laws_sim_v3.py                       # simulazione base (faker)   ║
║    python laws_sim_v3.py --real-yolo           # con YOLO       ║
║    python laws_sim_v3.py --demo-patch img.jpg  # demo adversarial    ║
║    python laws_sim_v3.py --demo-patch webcam   # demo da webcam      ║
║    python laws_sim_v3.py --steps 200 --verbose                       ║
║                                                                       ║
║  Dipendenze:                                                          ║
║    pip install ultralytics torch torchvision rich matplotlib faker   ║
║                                                                       ║
║  Peso modello    ║
║    yolov8n.pt  (~6MB, COCO, classe 0 = "person")                    ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import sys, math, json, random, argparse, os
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path

# ── Rich ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    console  = Console()
    HAS_RICH = True
except ImportError:
    class _FC:
        def print(self, *a, **kw): print(*[str(x) for x in a])
        def rule(self, *a, **kw):  print("─" * 60)
    console  = _FC()
    HAS_RICH = False

# ── Torch & Torchvision ───────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from   torchvision.transforms import ColorJitter
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    console.print("[yellow]⚠ torch non trovato — PatchOptimizer disabilitato[/yellow]")

# ── YOLO ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO  = False
    console.print("[yellow]⚠ ultralytics non trovato — usando VisionAgent analitico[/yellow]")

# ── OpenCV ────────────────────────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Faker ─────────────────────────────────────────────────────────────
try:
    from faker import Faker
    _fake    = Faker(["it_IT", "en_US"])
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

# ── PIL ───────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ══════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE GLOBALE
# ══════════════════════════════════════════════════════════════════════

GRID_SIZE          = 30
N_TARGETS          = 3
N_CIVILIANS        = 15
DRONE_HEIGHT       = 10.0
YOLO_MAX_RANGE     = 25.0
FUSION_WEIGHTS     = {"vision": 0.45, "osint": 0.35, "behavioral": 0.20}
ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD      = 0.38
TRACK_THRESHOLD      = 0.22

# ── Parametri VisionAgent analitico (fallback) ────────────────────────
PATCH_SUPPRESSION    = 0.65
PATCH_DIST_FALLOFF   = 0.04

# ── Parametri CLAE v2 ─────────────────────────────────────────────────
PATCH_BBOX_COVERAGE   = 0.28   # pixel patch / pixel bbox
OSINT_FIELDS_TOTAL    = 10
OSINT_FIELDS_POISONED = 3

# ── Parametri PatchOptimizer ─────────────────────────────────────────
PATCH_H         = 100      # altezza patch in pixel
PATCH_W         = 80       # larghezza patch in pixel
PATCH_LR        = 0.03     # learning rate Adam
PATCH_STEPS     = 80       # iterazioni ottimizzazione
PATCH_EPS       = 0.05     # clip perturbazione massima (0=nessun clip)
EOT_N_TRANSFORMS = 8       # trasformazioni per aspettativa EoT
PERSON_CLASS_ID  = 0       # classe COCO "person"
IMG_SIZE         = 640     # input size YOLOv8


# ══════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ══════════════════════════════════════════════════════════════════════

class AgentRole(Enum):
    CIVILIAN = "civilian"
    TARGET   = "target"

class AttackScenario(Enum):
    NONE         = "Baseline (No Attack)"
    PATCH_ONLY   = "Adversarial Patch [Vision]"
    OSINT_POISON = "OSINT Poisoning"
    CASCADING    = "Cascading Attack [Multi-Layer]"


@dataclass
class OSINTProfile:
    name               : str
    age                : int
    nationality        : str
    social_score       : float
    geo_anomaly        : float
    network_centrality : float
    is_poisoned        : bool = False

    @property
    def threat_score(self) -> float:
        return float(np.clip(
            0.40 * self.social_score +
            0.35 * self.geo_anomaly  +
            0.25 * self.network_centrality, 0, 1))


@dataclass
class VisionDetection:
    detected        : bool
    confidence      : float
    bbox            : Tuple[int, int, int, int]    # x, y, w, h
    class_label     : str
    patch_active    : bool  = False
    patch_coverage  : float = 0.0   # patch_area / bbox_area  [0-1]
    real_yolo       : bool  = False  # True se la confidence viene da YOLO reale


@dataclass
class FusionResult:
    threat_score        : float
    vision_contrib      : float
    osint_contrib       : float
    behavioral_contrib  : float
    confidence_interval : Tuple[float, float]


@dataclass
class Decision:
    action        : str
    threat_score  : float
    ihl_compliant : bool
    rationale     : str


@dataclass
class SimMetrics:
    tp            : int   = 0
    fp            : int   = 0
    tn            : int   = 0
    fn            : int   = 0
    threat_scores : List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp; return self.tp / d if d else 0.0
    @property
    def recall(self) -> float:
        d = self.tp + self.fn; return self.tp / d if d else 0.0
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2*p*r/(p+r) if (p+r) else 0.0
    @property
    def fpr(self) -> float:
        d = self.fp + self.tn; return self.fp / d if d else 0.0


# ══════════════════════════════════════════════════════════════════════
# ADVERSARIAL PATCH OPTIMIZER (EoT — Athalye et al. 2018)
# ══════════════════════════════════════════════════════════════════════

class PatchOptimizer:
    """
    Genera una adversarial patch ottimizzata tramite torch autograd + EoT.

    Obiettivo
    ─────────
    Minimizzare la confidence di YOLO sulla classe "person" (class 0 COCO)
    quando la patch viene applicata sopra il bounding box del target.

    Loss Function
    ─────────────
    Per ogni iterazione, invece di ottimizzare su un'immagine fissa,
    applichiamo N trasformazioni fisiche casuali (EoT):

        L_EoT = (1/N) Σ_{t~T} L(f(apply_patch(t(img), t(patch))), y_bg)

    dove:
        f         = modello YOLO (forward pass)
        t         = trasformazione casuale da T = {rotate, scale, jitter, flip}
        y_bg      = target "background" (vogliamo che YOLO non veda persone)
        apply_patch(img, patch) = incolla patch sul bbox nella regione target

    La loss usata è la negative log-likelihood della classe "person":
    massimizziamo l'entropia sulle classi, spingendo la distribuzione
    di output lontano dalla classe 0 (person).

    Questo è equivalente al targeted attack con target = "background":
        L = -log(1 - P(person | img_patch))

    EoT Transforms T
    ─────────────────
    T1: rotazione casuale in [-20°, +20°]     (drone non perfettamente allineato)
    T2: scala casuale in [0.75, 1.25]          (variazione altitudine/distanza)
    T3: color jitter (brightness, contrast)    (condizioni di illuminazione)
    T4: flip orizzontale con p=0.5            (orientamento soggetto)
    T5: blur gaussiano σ in [0, 1.5]           (defocus, mosso)

    Queste trasformazioni simulano le condizioni fisiche del C.A.R.E. Kit:
    la patch è stampata su tessuto e vista da angoli/distanze variabili.
    """

    def __init__(self, patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 model_path: str = "yolov8n.pt"):
        if not HAS_TORCH:
            raise RuntimeError("torch non disponibile")
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.model_path = model_path
        self._yolo      = None

        # Patch inizializzata con rumore casuale uniforme in [0,1]
        # Sarà ottimizzata verso valori che confondono YOLO
        self.patch = torch.rand(3, patch_h, patch_w,
                                dtype=torch.float32,
                                requires_grad=True)

    # ── YOLO model (lazy load) ────────────────────────────────────────
    def _get_model(self):
        if self._yolo is None:
            if not HAS_YOLO:
                raise RuntimeError("ultralytics non disponibile")
            self._yolo = _YOLO(self.model_path)
            # Congela tutti i parametri del modello: ottimizziamo SOLO la patch
            for p in self._yolo.model.parameters():
                p.requires_grad_(False)
            self._yolo.model.eval()
        return self._yolo

    # ── EoT: singola trasformazione casuale ───────────────────────────
    @staticmethod
    def _random_transform(patch: "torch.Tensor") -> "torch.Tensor":
        """
        Applica una trasformazione fisica casuale alla patch.
        Tutte le operazioni sono differenziabili rispetto a patch.
        """
        t = patch.unsqueeze(0)   # [1, C, H, W]

        # T1: rotazione (simula angolo drone)
        angle = random.uniform(-20, 20)
        t = TF.rotate(t, angle)

        # T2: scala (simula distanza)
        scale = random.uniform(0.75, 1.25)
        new_h = max(1, int(patch.shape[1] * scale))
        new_w = max(1, int(patch.shape[2] * scale))
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                          align_corners=False)
        # Riporta alle dimensioni originali
        t = F.interpolate(t, size=(patch.shape[1], patch.shape[2]),
                          mode="bilinear", align_corners=False)

        # T3: color jitter (illuminazione)
        brightness = random.uniform(0.7, 1.3)
        contrast   = random.uniform(0.8, 1.2)
        t = TF.adjust_brightness(t, brightness)
        t = TF.adjust_contrast(t, contrast)

        # T4: flip orizzontale (orientamento soggetto)
        if random.random() < 0.5:
            t = TF.hflip(t)

        # T5: blur gaussiano (defocus, movimento)
        sigma = random.uniform(0, 1.5)
        if sigma > 0.3:
            k = int(sigma * 4) | 1  # kernel size dispari
            k = max(3, k)
            t = TF.gaussian_blur(t, kernel_size=k, sigma=sigma)

        return torch.clamp(t.squeeze(0), 0, 1)

    # ── Applica patch su immagine tensor ─────────────────────────────
    @staticmethod
    def apply_patch_to_image(img_t: "torch.Tensor",
                             patch_t: "torch.Tensor",
                             bbox: Tuple[int, int, int, int]) -> "torch.Tensor":
        """
        Incolla la patch sul tensore dell'immagine nella regione bbox.

        img_t  : [3, H, W] float32 normalizzato [0,1]
        patch_t: [3, ph, pw] float32 [0,1]
        bbox   : (x1, y1, x2, y2) in pixel

        Questa è la "tensor slicing" operation che sostituisce i pixel
        del target con i pixel della patch. Nel C.A.R.E. Kit fisico,
        questa operazione corrisponde all'applicazione fisica del pattern
        sull'indumento del soggetto.

        Nota: l'operazione è differenziabile — il gradiente fluisce
        attraverso i pixel della regione incollata verso patch_t.
        """
        x1, y1, x2, y2 = bbox
        ph = y2 - y1
        pw = x2 - x1
        if ph <= 0 or pw <= 0:
            return img_t
        # Ridimensiona patch al bbox
        patch_resized = F.interpolate(patch_t.unsqueeze(0),
                                      size=(ph, pw),
                                      mode="bilinear",
                                      align_corners=False).squeeze(0)
        # Clone necessario per poter fare in-place assignment con autograd
        img_clone = img_t.clone()
        img_clone[:, y1:y2, x1:x2] = patch_resized
        return img_clone

    # ── Loss: negative log-likelihood della classe "person" ──────────
    @staticmethod
    def _patch_loss(predictions, person_class: int = PERSON_CLASS_ID) -> "torch.Tensor":
        """
        Data la lista di predizioni YOLO (tensore delle confidence per ogni box),
        calcola la loss per massimizzare la probabilità di NON rilevare persone.

        predictions: lista di Result objects da ultralytics
        Estraggo le confidence della classe "person" e calcolo:

            L = mean(conf_person)   →   vogliamo minimizzare questo

        Equivalente a: massimizzare la probabilità che YOLO classifichi
        tutti i bounding box come "non-person".

        In una formulazione più rigida si userebbe:
            L = -log(1 - max(conf_person))
        ma la versione mean è più stabile numericamente e produce
        gradients più utili per l'ottimizzazione della patch.
        """
        all_conf = []
        for result in predictions:
            if result.boxes is not None and len(result.boxes) > 0:
                # boxes.cls contiene le classi predette
                # boxes.conf contiene le confidence
                person_mask = (result.boxes.cls == person_class)
                if person_mask.any():
                    all_conf.append(result.boxes.conf[person_mask])
        if all_conf:
            return torch.cat(all_conf).mean()
        # Se YOLO non rileva nessuna persona → loss 0 (già ingannato)
        return torch.tensor(0.0, requires_grad=True)

    # ── Metodo principale: ottimizzazione EoT ────────────────────────
    def optimize(self, image_path: str,
                 bbox: Optional[Tuple[int, int, int, int]] = None,
                 n_steps: int = PATCH_STEPS,
                 lr: float    = PATCH_LR,
                 n_eot: int   = EOT_N_TRANSFORMS,
                 verbose: bool = True) -> dict:
        """
        Ottimizza la patch su image_path con EoT.

        Parametri
        ─────────
        image_path : percorso immagine sorgente (o "webcam")
        bbox       : (x1,y1,x2,y2) dove applicare la patch.
                     Se None, usa il bbox della prima persona rilevata.
        n_steps    : iterazioni di ottimizzazione
        lr         : learning rate Adam
        n_eot      : numero trasformazioni EoT per step
        verbose    : stampa loss ogni 10 step

        Returns
        ───────
        dict con:
            "patch"         : tensore patch ottimizzata [3,H,W]
            "conf_before"   : confidence YOLO prima dell'attacco
            "conf_after"    : confidence YOLO dopo l'attacco
            "loss_history"  : lista loss per step
            "patch_coverage": patch_area / bbox_area (per CLAE)
        """
        model = self._get_model()
        if not HAS_PIL:
            raise RuntimeError("PIL non disponibile")

        # ── Carica immagine ──────────────────────────────────────────
        if image_path == "webcam":
            if not HAS_CV2:
                raise RuntimeError("OpenCV richiesto per webcam")
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Webcam non disponibile")
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.open(image_path).convert("RGB")

        # Ridimensiona a IMG_SIZE (come YOLO)
        img_pil  = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_np   = np.array(img_pil).astype(np.float32) / 255.0
        img_t    = torch.from_numpy(img_np).permute(2, 0, 1)  # [3,H,W]

        # ── Rilevamento baseline (senza patch) ───────────────────────
        console.print("[cyan]Rilevamento baseline (senza patch)...[/cyan]")
        results_base = model(img_pil, verbose=False)
        conf_before  = self._get_person_conf(results_base)
        console.print(f"  Confidence person PRIMA patch: [bold]{conf_before:.4f}[/bold]")

        if conf_before < 0.01:
            console.print("[yellow]⚠ Nessuna persona rilevata nell'immagine baseline[/yellow]")

        # ── Trova bbox persona (se non fornito) ───────────────────────
        if bbox is None:
            bbox = self._find_person_bbox(results_base)
        if bbox is None:
            # Bbox di default: centro immagine
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            bbox = (cx-60, cy-100, cx+60, cy+100)

        x1, y1, x2, y2 = bbox
        bbox_area  = (x2-x1) * (y2-y1)
        patch_area = self.patch_h * self.patch_w
        coverage   = patch_area / max(bbox_area, 1)

        console.print(f"  BBox: {bbox}  (area={bbox_area}px)")
        console.print(f"  Patch: {self.patch_h}×{self.patch_w}  "
                      f"coverage={coverage:.3f}  ({coverage*100:.1f}% del bbox)")

        # ── Ottimizzazione EoT ────────────────────────────────────────
        optimizer    = torch.optim.Adam([self.patch], lr=lr)
        loss_history = []

        console.print(f"\n[cyan]Ottimizzazione EoT: {n_steps} step, "
                      f"{n_eot} trasf./step...[/cyan]")

        for step in range(n_steps):
            optimizer.zero_grad()
            step_losses = []

            for _ in range(n_eot):
                # Trasforma patch (simula variazioni fisiche)
                patch_t = self._random_transform(self.patch)
                # Applica su immagine (tensore con gradiente verso patch)
                img_patched = self.apply_patch_to_image(img_t, patch_t, bbox)
                # Prepara batch per YOLO: shape (1, 3, H, W) in float32 [0,1]
                batch = img_patched.unsqueeze(0)  # aggiunge dimensione batch
                # Forward pass YOLO (senza no_grad, traccia il grafo)
                results = model(batch, verbose=False)
                # Loss: media confidence person su questa trasformazione
                conf_tensor = self._get_person_conf_tensor(results)
                step_losses.append(conf_tensor)

            # Loss finale = media su tutte le trasformazioni EoT
            loss = torch.stack(step_losses).mean() if step_losses else \
                   torch.tensor(0.0, requires_grad=True)
            loss.backward()
            optimizer.step()

            # Clamp patch in [0,1]
            with torch.no_grad():
                self.patch.clamp_(0, 1)

            loss_val = loss.item()
            loss_history.append(loss_val)

            if verbose and (step % 10 == 0 or step == n_steps-1):
                console.print(f"  Step {step+1:3d}/{n_steps}  loss={loss_val:.4f}")

        # ── Valutazione finale ─────────────────────────────────────────
        console.print("\n[cyan]Valutazione con patch ottimizzata...[/cyan]")
        patch_final_np  = (self.patch.detach().permute(1,2,0)
                           .numpy() * 255).astype(np.uint8)
        img_final       = self.apply_patch_to_image(
                              img_t, self.patch.detach(), bbox)
        img_final_np    = (img_final.permute(1,2,0).numpy()*255).astype(np.uint8)
        img_final_pil   = Image.fromarray(img_final_np)

        results_after   = model(img_final_pil, verbose=False)
        conf_after      = self._get_person_conf(results_after)

        console.print(f"  Confidence person DOPO patch:  [bold red]{conf_after:.4f}[/bold red]")
        drop = conf_before - conf_after
        console.print(f"  Drop: [bold yellow]{drop:+.4f}  "
                      f"({drop/max(conf_before,1e-6)*100:.1f}%)[/bold yellow]")

        return {
            "patch"           : self.patch.detach(),
            "conf_before"     : conf_before,
            "conf_after"      : conf_after,
            "conf_drop"       : drop,
            "loss_history"    : loss_history,
            "patch_coverage"  : coverage,
            "bbox"            : bbox,
            "img_original"    : img_t,
            "img_patched"     : img_final,
            "patch_arr"       : patch_final_np,
        }

    # ── Utility ───────────────────────────────────────────────────────
    @staticmethod
    def _get_person_conf(results) -> float:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0

    @staticmethod
    def _get_person_conf_tensor(results) -> "torch.Tensor":
        """
        Estrae la confidence media della classe 'person' come tensore scalare
        mantenendo il collegamento al grafo computazionale.
        """
        # results è una lista di oggetti Results di ultralytics quando l'input è un tensore.
        # I tensori interni (boxes.conf, boxes.cls) conservano la storia dei gradienti.
        conf_list = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    conf_list.append(r.boxes.conf[mask])
        if conf_list:
            # Concatena e calcola la media (scalare differenziabile)
            all_conf = torch.cat(conf_list)
            return all_conf.mean()
        # Se nessuna persona rilevata, restituisci un tensore zero che richiede grad
        return torch.tensor(0.0, device=results[0].boxes.conf.device if results else 'cpu',
                            requires_grad=True)

    @staticmethod
    def _find_person_bbox(results) -> Optional[Tuple[int,int,int,int]]:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    box = r.boxes.xyxy[mask][0].int().tolist()
                    return tuple(box)
        return None

    # ── Salva output visivi ────────────────────────────────────────────
    @staticmethod
    def save_results(result_dict: dict, output_dir: str = "."):
        if not HAS_MPL:
            return
        out = Path(output_dir)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Adversarial Patch — Risultati EoT", fontsize=13, fontweight="bold")

        # Immagine originale
        orig_np = (result_dict["img_original"].permute(1,2,0).numpy()*255).astype(np.uint8)
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Originale\nconf person = {result_dict['conf_before']:.4f}",
                          color="green")
        x1,y1,x2,y2 = result_dict["bbox"]
        rect = mpatches.Rectangle((x1,y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor="green",
                                   facecolor="none")
        axes[0].add_patch(rect)
        axes[0].axis("off")

        # Patch
        axes[1].imshow(result_dict["patch_arr"])
        cov = result_dict["patch_coverage"]
        axes[1].set_title(f"Patch ottimizzata (EoT)\ncoverage = {cov:.3f} → C_vision",
                          color="orange")
        axes[1].axis("off")

        # Immagine con patch applicata
        patched_np = (result_dict["img_patched"].permute(1,2,0).numpy()*255).astype(np.uint8)
        axes[2].imshow(patched_np)
        axes[2].set_title(f"Con patch applicata\nconf person = {result_dict['conf_after']:.4f}",
                          color="red")
        rect2 = mpatches.Rectangle((x1,y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor="red",
                                    facecolor="none")
        axes[2].add_patch(rect2)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out / "patch_result.png", dpi=150, bbox_inches="tight")

        # Loss curve
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result_dict["loss_history"], color="#e74c3c", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step ottimizzazione")
        ax.set_ylabel("Loss (confidence person media)")
        ax.set_title("Convergenza EoT — Adversarial Patch Loss")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "patch_loss.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        console.print(f"[green]Plot salvati → {out}/patch_result.png, patch_loss.png[/green]")


# ══════════════════════════════════════════════════════════════════════
# VISION AGENT REALE (YOLOv8 + fallback analitico)
# ══════════════════════════════════════════════════════════════════════

class VisionAgentReal:
    """
    VisionAgent con YOLO reale.

    Modalità
    ────────
    real_mode=True  : usa YOLOv8 su frame reali (VisDrone / webcam)
                      Richiede: ultralytics, frame PIL/numpy
    real_mode=False : fallback al modello analitico v2 (formulazione EoT)

    Nel real_mode, la confidence viene dal forward pass di YOLO
    sull'immagine corrente (con o senza patch applicata al tensore).

    Nel fallback, la confidence è modellata analiticamente:
        conf_base(d) = 0.95 * exp(-1.5 * d/D_max) + ε
        conf_patch   = conf_base * (1 - S_eff(d) + ε_phys)

    Come collegare a VisDrone
    ─────────────────────────
    Per usare frame VisDrone reali:
        1. Scarica VisDrone2019-DET: https://github.com/VisDrone/VisDrone-Dataset
        2. Passa image_dir= al costruttore
        3. Il simulatore carica frame in sequenza a ogni step

    Passaggio da simulato a reale
    ──────────────────────────────
    # ATTUALE (simulato)
    vision_d = agent.detect(entity, distance, patch_active=True)

    # REALE (YOLOv8 su frame)
    frame = load_frame("visdrone/images/train/0000001_00001_d_0000001.jpg")
    vision_d = agent.detect_real(frame, apply_patch=True, patch_tensor=patch)
    """

    DETECTION_THRESHOLD = 0.25

    def __init__(self, real_mode: bool = False,
                 model_path: str = "yolov8n.pt",
                 image_dir:  Optional[str] = None,
                 patch_tensor: Optional["torch.Tensor"] = None):
        self.real_mode    = real_mode and HAS_YOLO and HAS_TORCH
        self.model_path   = model_path
        self.image_dir    = image_dir
        self.patch_tensor = patch_tensor
        self._model       = None
        self._frames      = []
        self._frame_idx   = 0
        self.fn_count     = 0
        self.det_count    = 0

        if self.real_mode:
            self._load_model()
            if image_dir:
                self._index_frames()
        elif real_mode and not HAS_YOLO:
            console.print("[yellow]⚠ real_mode richiesto ma ultralytics non disponibile "
                          "→ fallback analitico[/yellow]")

    def _load_model(self):
        try:
            self._model = _YOLO(self.model_path)
            for p in self._model.model.parameters():
                p.requires_grad_(False)
            self._model.model.eval()
            console.print(f"[green]✓ YOLOv8 caricato: {self.model_path}[/green]")
        except Exception as e:
            console.print(f"[red]Errore caricamento YOLO: {e} → fallback analitico[/red]")
            self.real_mode = False

    def _index_frames(self):
        exts = {".jpg", ".jpeg", ".png"}
        self._frames = sorted(
            [f for f in Path(self.image_dir).rglob("*") if f.suffix.lower() in exts])
        console.print(f"[green]✓ Indicizzati {len(self._frames)} frame da {self.image_dir}[/green]")

    def _next_frame(self) -> Optional["Image.Image"]:
        if not self._frames:
            return None
        f = self._frames[self._frame_idx % len(self._frames)]
        self._frame_idx += 1
        return Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    # ── Detection su entità simulata (fallback analitico) ────────────
    def detect(self, entity, distance: float,
               patch_active: bool = False) -> VisionDetection:
        """
        Rilevamento su entità nella simulazione.
        Se real_mode e ci sono frame disponibili, usa YOLO su frame reale.
        Altrimenti usa il modello analitico.
        """
        self.det_count += 1

        if self.real_mode and self._frames:
            return self._detect_real_from_sim(entity, distance, patch_active)
        return self._detect_analytical(entity, distance, patch_active)

    def _detect_analytical(self, entity, distance: float,
                            patch_active: bool) -> VisionDetection:
        """Modello analitico EoT — identico a v2."""
        norm  = distance / YOLO_MAX_RANGE
        base  = float(np.clip(0.95 * math.exp(-1.5 * norm)
                              + random.gauss(0, 0.04), 0, 1))

        if patch_active and entity.care_kit_active:
            S_eff = PATCH_SUPPRESSION * math.exp(-PATCH_DIST_FALLOFF * distance)
            conf  = float(np.clip(base * (1.0 - S_eff + random.gauss(0, 0.08)), 0, 1))
            cov   = PATCH_BBOX_COVERAGE
        else:
            conf = base
            cov  = 0.0

        detected = conf >= self.DETECTION_THRESHOLD
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected       = detected,
            confidence     = conf,
            bbox           = (entity.x, entity.y, 5, 8),
            class_label    = "person" if detected else "background",
            patch_active   = patch_active and entity.care_kit_active,
            patch_coverage = cov,
            real_yolo      = False,
        )

    def _detect_real_from_sim(self, entity, distance: float,
                               patch_active: bool) -> VisionDetection:
        """
        Usa frame reale ma mappa la confidence alla distanza simulata.

        Flusso:
        1. Carica il prossimo frame VisDrone
        2. Se patch attiva, applica il patch_tensor sul frame
        3. Esegue YOLO e ottiene la confidence
        4. Usa la confidence reale ma la scala con la distanza simulata
           (perché nel sim non sappiamo dove esattamente si trova l'entità
           nel frame reale → scaling geometrico come correzione)

        Nel lavoro futuro: sostituire con tracking reale dell'entità
        nel frame (associando bbox YOLO a entità simulate tramite posizione).
        """
        frame = self._next_frame()
        if frame is None:
            return self._detect_analytical(entity, distance, patch_active)

        img_t = torch.from_numpy(
            np.array(frame).astype(np.float32) / 255.0).permute(2, 0, 1)

        apply_p = patch_active and entity.care_kit_active and \
                  self.patch_tensor is not None

        if apply_p:
            # Bbox default: regione centrale del frame (placeholder)
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            bbox_px = (cx-50, cy-80, cx+50, cy+80)
            img_t = PatchOptimizer.apply_patch_to_image(
                img_t, self.patch_tensor, bbox_px)
            cov = (self.patch_tensor.shape[1] * self.patch_tensor.shape[2]) / \
                  max((100 * 160), 1)
        else:
            cov = 0.0

        img_pil  = Image.fromarray((img_t.permute(1,2,0).numpy()*255).astype(np.uint8))
        results  = self._model(img_pil, verbose=False)

        # Confidence reale da YOLO
        yolo_conf = PatchOptimizer._get_person_conf(results)

        # Scala con distanza (lontananza riduce la rilevabilità anche nel reale)
        dist_scale = math.exp(-1.0 * distance / YOLO_MAX_RANGE)
        conf = float(np.clip(yolo_conf * dist_scale + random.gauss(0, 0.02), 0, 1))

        detected = conf >= self.DETECTION_THRESHOLD
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected       = detected,
            confidence     = conf,
            bbox           = (entity.x, entity.y, 5, 8),
            class_label    = "person" if detected else "background",
            patch_active   = apply_p,
            patch_coverage = cov,
            real_yolo      = True,
        )

    # ── Detection standalone su immagine (per demo e test diretti) ───
    def detect_image(self, image_path: str,
                     apply_patch: bool = False,
                     patch_tensor: Optional["torch.Tensor"] = None,
                     bbox: Optional[Tuple] = None) -> Dict:
        """
        Detection standalone su immagine reale.
        Usato dal demo mode e dai test unitari del VisionAgent.

        Esempio:
            agent = VisionAgentReal(real_mode=True)
            result = agent.detect_image("visdrone/frame.jpg")
            print(result["confidence"], result["boxes"])
        """
        if not self.real_mode:
            raise RuntimeError("detect_image richiede real_mode=True")

        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_t = torch.from_numpy(
            np.array(img).astype(np.float32)/255.0).permute(2,0,1)

        if apply_patch and patch_tensor is not None:
            if bbox is None:
                results_pre = self._model(img, verbose=False)
                bbox = PatchOptimizer._find_person_bbox(results_pre) or \
                       (IMG_SIZE//2-50, IMG_SIZE//2-80, IMG_SIZE//2+50, IMG_SIZE//2+80)
            img_t = PatchOptimizer.apply_patch_to_image(img_t, patch_tensor, bbox)
            img   = Image.fromarray((img_t.permute(1,2,0).numpy()*255).astype(np.uint8))

        results = self._model(img, verbose=False)
        boxes   = []
        for r in results:
            if r.boxes is not None:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    boxes.append({
                        "bbox"  : box.int().tolist(),
                        "class" : int(cls),
                        "label" : r.names[int(cls)],
                        "conf"  : float(conf),
                    })
        person_conf = max((b["conf"] for b in boxes if b["class"] == PERSON_CLASS_ID),
                          default=0.0)
        return {
            "boxes"      : boxes,
            "confidence" : person_conf,
            "n_persons"  : sum(1 for b in boxes if b["class"] == PERSON_CLASS_ID),
        }


# ══════════════════════════════════════════════════════════════════════
# AGENTS (OSINT, FUSION, DECISION) — identici a v2
# ══════════════════════════════════════════════════════════════════════

class SimEntity:
    def __init__(self, eid, role, grid):
        self.id              = eid
        self.role            = role
        self.grid            = grid
        self.x               = random.randint(0, grid - 1)
        self.y               = random.randint(0, grid - 1)
        self.history         = [(self.x, self.y)]
        self.care_kit_active = False
        self.osint_profile   = self._gen_profile()

    def _gen_profile(self):
        name = _fake.name() if HAS_FAKER else f"Entity_{self.id:03d}"
        nat  = _fake.country() if HAS_FAKER else "N/A"
        if self.role == AgentRole.TARGET:
            return OSINTProfile(name=name, age=random.randint(20,45), nationality=nat,
                social_score=random.uniform(0.55,0.90),
                geo_anomaly=random.uniform(0.50,0.85),
                network_centrality=random.uniform(0.40,0.80))
        return OSINTProfile(name=name, age=random.randint(18,70), nationality=nat,
            social_score=random.uniform(0.00,0.30),
            geo_anomaly=random.uniform(0.00,0.25),
            network_centrality=random.uniform(0.00,0.20))

    def move(self):
        step = 2 if self.role == AgentRole.TARGET else 1
        self.x = int(np.clip(self.x + random.randint(-step,step), 0, self.grid-1))
        self.y = int(np.clip(self.y + random.randint(-step,step), 0, self.grid-1))
        self.history.append((self.x, self.y))
        if len(self.history) > 20: self.history.pop(0)

    @property
    def behavioral_score(self):
        if len(self.history) < 3: return 0.0
        arr = np.array(self.history[-10:])
        sc  = np.clip((np.var(arr[:,0]) + np.var(arr[:,1])) / 20.0, 0, 1)
        if self.role == AgentRole.TARGET:
            sc = float(np.clip(sc + random.uniform(0.10, 0.25), 0, 1))
        return float(sc)


class Environment:
    def __init__(self, grid=GRID_SIZE):
        self.grid    = grid; self.step_idx = 0
        self.drone_x = grid//2; self.drone_y = grid//2
        self.entities: List[SimEntity] = []
        for i in range(N_TARGETS):
            self.entities.append(SimEntity(i, AgentRole.TARGET, grid))
        for i in range(N_CIVILIANS):
            self.entities.append(SimEntity(N_TARGETS+i, AgentRole.CIVILIAN, grid))

    def tick(self):
        for e in self.entities: e.move()
        self.drone_x = int(np.clip(self.drone_x+random.randint(-2,2), 0, self.grid-1))
        self.drone_y = int(np.clip(self.drone_y+random.randint(-2,2), 0, self.grid-1))
        self.step_idx += 1

    def dist_to_drone(self, e):
        return math.sqrt((e.x-self.drone_x)**2 + (e.y-self.drone_y)**2 + DRONE_HEIGHT**2)

    def visible(self):
        return [e for e in self.entities if self.dist_to_drone(e) <= YOLO_MAX_RANGE]

    def nearby_civilians(self, t, r=5):
        return sum(1 for e in self.entities if e.role == AgentRole.CIVILIAN
                   and abs(e.x-t.x)+abs(e.y-t.y) <= r)


class OSINTAgent:
    def __init__(self): self.cache = {}

    def analyze(self, entity, poisoning=False):
        p = deepcopy(entity.osint_profile)
        if poisoning and entity.role == AgentRole.TARGET:
            p.social_score       = max(0.0, p.social_score - random.uniform(0.40,0.60))
            p.geo_anomaly        = max(0.0, p.geo_anomaly  - random.uniform(0.30,0.55))
            p.network_centrality = max(0.0, p.network_centrality - random.uniform(0.25,0.45))
            p.is_poisoned = True
        self.cache[entity.id] = p
        return p


class FusionAgent:
    PRIOR = 0.50
    def __init__(self):
        self.history = {}
        self.w = FUSION_WEIGHTS

    def _bay(self, prior, lik):
        p = lik*prior; q = (1-lik)*(1-prior)
        return float(np.clip(p/(p+q+1e-9), 0, 1))

    def _ci(self, scores):
        if len(scores) < 2: return (0.0, 1.0)
        arr = np.array(scores)
        return (max(0.0,float(np.mean(arr)-1.96*np.std(arr))),
                min(1.0,float(np.mean(arr)+1.96*np.std(arr))))

    def fuse(self, eid, vision, osint, behavioral):
        v = vision.confidence * (1.0 if vision.detected else 0.5)
        raw = self.w["vision"]*v + self.w["osint"]*osint.threat_score + \
              self.w["behavioral"]*behavioral
        threat = self._bay(self.PRIOR, raw)
        self.history.setdefault(eid, [])
        self.history[eid].append(threat)
        if len(self.history[eid]) > 20: self.history[eid].pop(0)
        return FusionResult(
            threat_score=threat,
            vision_contrib=self.w["vision"]*v,
            osint_contrib=self.w["osint"]*osint.threat_score,
            behavioral_contrib=self.w["behavioral"]*behavioral,
            confidence_interval=self._ci(self.history[eid][-10:]))


class DecisionAgent:
    def __init__(self):
        self.log=[]; self.engagements=0; self.ihl_overrides=0

    def _ihl_ok(self, fusion, civs):
        if (fusion.confidence_interval[1]-fusion.confidence_interval[0]) > 0.40:
            return False
        if civs > 3 and fusion.threat_score < 0.85: return False
        return True

    def decide(self, eid, fusion, nearby_civs=0):
        ihl    = self._ihl_ok(fusion, nearby_civs)
        score  = fusion.threat_score
        if   score >= ENGAGEMENT_THRESHOLD and ihl: action = "ENGAGE"; self.engagements+=1
        elif score >= ALERT_THRESHOLD:               action = "ALERT"
        elif score >= TRACK_THRESHOLD:               action = "TRACK"
        else:                                        action = "IGNORE"
        if action=="ENGAGE" and not ihl: action="ALERT"; self.ihl_overrides+=1
        rat = (f"score={score:.3f} CI=[{fusion.confidence_interval[0]:.2f},"
               f"{fusion.confidence_interval[1]:.2f}] "
               f"V={fusion.vision_contrib:.3f} O={fusion.osint_contrib:.3f} "
               f"B={fusion.behavioral_contrib:.3f}")
        d = Decision(action=action,threat_score=score,ihl_compliant=ihl,rationale=rat)
        self.log.append(d)
        return d


# ══════════════════════════════════════════════════════════════════════
# CLAE v2 — Costi fisicamente misurabili
# ══════════════════════════════════════════════════════════════════════

def clae_costs() -> Dict[str, Optional[float]]:
    """
    Calcola i costi CLAE da grandezze fisiche misurabili.

    C_vision   = patch_area / bbox_area
                 Nel YOLO reale: misurato in pixel effettivi
                 Nella simulazione: PATCH_BBOX_COVERAGE (parametro fisso)

    C_osint    = fields_poisoned / fields_total
                 Frazione di voci OSINT falsificate dall'avversario

    C_cascading = 1 - (1 - C_v)(1 - C_o)
                 Probabilità unione: i due attacchi condividono
                 overhead di pianificazione → costo totale < additivo
    """
    c_v = PATCH_BBOX_COVERAGE
    c_o = OSINT_FIELDS_POISONED / OSINT_FIELDS_TOTAL
    c_c = 1.0 - (1.0 - c_v) * (1.0 - c_o)
    return {"NONE": None, "PATCH_ONLY": c_v,
            "OSINT_POISON": c_o, "CASCADING": c_c}


def compute_clae(scenario, attack_m, baseline_m) -> Optional[float]:
    """
    CLAE = ΔF1 / C_attack

    Differenza v1→v2: C non è più un numero arbitrario ma deriva
    da grandezze misurabili (pixel ratio e OSINT field ratio).
    Questo rende la metrica confrontabile tra attacchi fisici e digitali.
    """
    costs = clae_costs()
    cost  = costs.get(scenario.name)
    if cost is None: return None
    return (baseline_m.f1 - attack_m.f1) / cost if cost > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# CORE SIMULATOR
# ══════════════════════════════════════════════════════════════════════

class LAWSSim:
    def __init__(self, scenario, steps, seed=42,
                 real_mode=False, image_dir=None, patch_tensor=None):
        random.seed(seed); np.random.seed(seed)
        self.scenario = scenario
        self.steps    = steps
        self.env      = Environment()
        self.osint    = OSINTAgent()
        self.vision   = VisionAgentReal(
            real_mode    = real_mode,
            image_dir    = image_dir,
            patch_tensor = patch_tensor,
        )
        self.fusion   = FusionAgent()
        self.decision = DecisionAgent()
        self.metrics  = SimMetrics()
        self.step_log = []

        if scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING):
            for e in self.env.entities:
                if e.role == AgentRole.TARGET:
                    e.care_kit_active = True

    def run(self, verbose=False):
        patch_on  = self.scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING)
        poison_on = self.scenario in (AttackScenario.OSINT_POISON, AttackScenario.CASCADING)

        for _ in range(self.steps):
            self.env.tick()
            for entity in self.env.visible():
                dist     = self.env.dist_to_drone(entity)
                civ_near = self.env.nearby_civilians(entity)
                osint_p  = self.osint.analyze(entity, poisoning=poison_on)
                vision_d = self.vision.detect(entity, dist, patch_active=patch_on)
                fusion_r = self.fusion.fuse(entity.id, vision_d, osint_p,
                                            entity.behavioral_score)
                decision = self.decision.decide(entity.id, fusion_r, civ_near)

                is_threat  = entity.role == AgentRole.TARGET
                is_engaged = decision.action in ("ENGAGE", "ALERT")
                if   is_threat and     is_engaged: self.metrics.tp += 1
                elif not is_threat and is_engaged: self.metrics.fp += 1
                elif is_threat and not is_engaged: self.metrics.fn += 1
                else:                              self.metrics.tn += 1
                self.metrics.threat_scores.append(fusion_r.threat_score)

                if verbose and entity.role == AgentRole.TARGET:
                    self.step_log.append({
                        "step": self.env.step_idx, "id": entity.id,
                        "action": decision.action,
                        "score": round(fusion_r.threat_score, 4),
                        "patch": vision_d.patch_active,
                        "real_yolo": vision_d.real_yolo,
                        "poisoned": osint_p.is_poisoned,
                    })
        return self.metrics


# ══════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════

def print_results(results, baseline):
    if not HAS_RICH:
        for sc, m in results.items():
            clae = compute_clae(sc, m, baseline)
            print(f"  {sc.value:<42} F1={m.f1:.3f}  "
                  f"CLAE={'—' if clae is None else f'{clae:.3f}'}")
        return
    t = Table(show_header=True, header_style="bold magenta",
              title="[bold cyan]LAWS-SIM v3 — Risultati[/bold cyan]")
    for col in ["Scenario","Precision","Recall","F1","FPR","CLAE v2"]:
        t.add_column(col, justify="right" if col!="Scenario" else "left")
    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        cs   = "—" if clae is None else f"{clae:.3f}"
        col  = "green" if m.f1>0.60 else ("yellow" if m.f1>0.35 else "red")
        t.add_row(sc.value, f"{m.precision:.3f}", f"{m.recall:.3f}",
                  f"[{col}]{m.f1:.3f}[/{col}]", f"{m.fpr:.3f}", cs)
    console.print(t)

    costs = clae_costs()
    c_v,c_o,c_c = costs["PATCH_ONLY"], costs["OSINT_POISON"], costs["CASCADING"]
    console.print(Panel(
        f"[bold]CLAE v2 — derivazione costi misurabili[/bold]\n\n"
        f"C_vision   = patch_area/bbox_area = PATCH_BBOX_COVERAGE = [cyan]{c_v:.2f}[/cyan]\n"
        f"           → nel YOLO reale: pixel patch / pixel bbox\n\n"
        f"C_osint    = {OSINT_FIELDS_POISONED}/{OSINT_FIELDS_TOTAL} campi OSINT alterati "
        f"= [cyan]{c_o:.2f}[/cyan]\n"
        f"           → un profilo reale ha ~10 campi verificabili\n\n"
        f"C_cascade  = 1−(1−{c_v})(1−{c_o}) = [cyan]{c_c:.4f}[/cyan]  "
        f"[yellow](vs additivo {c_v+c_o:.2f} → −{(c_v+c_o-c_c)/(c_v+c_o)*100:.1f}%)[/yellow]\n"
        f"           → formula probabilità unione: overhead condiviso tra i due vettori",
        title="[bold yellow]CLAE v2[/bold yellow]", border_style="yellow"
    ))


def plot_results(results):
    if not HAS_MPL: return
    from matplotlib.gridspec import GridSpec
    scenarios = list(results.keys())
    labels    = [s.value.replace(" [","\n[") for s in scenarios]
    COLORS    = ["#2ecc71","#e74c3c","#f39c12","#9b59b6"]
    f1s  = [m.f1 for m in results.values()]
    prec = [m.precision for m in results.values()]
    rec  = [m.recall for m in results.values()]
    fprs = [m.fpr for m in results.values()]
    fig  = plt.figure(figsize=(17,10))
    fig.suptitle("LAWS-SIM v3 — Analisi Vulnerabilità Multi-Layer",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
    ax1 = fig.add_subplot(gs[0,:2])
    bars = ax1.bar(labels, f1s, color=COLORS, edgecolor="black", linewidth=0.6)
    ax1.axhline(f1s[0],color="#2ecc71",linestyle="--",alpha=0.6,
                label=f"Baseline F1={f1s[0]:.3f}")
    for bar,v in zip(bars,f1s):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                 f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax1.set_ylim(0,1.1); ax1.set_ylabel("F1 Score",fontsize=11)
    ax1.set_title("F1 Score per Scenario di Attacco",fontweight="bold"); ax1.legend()
    ax2 = fig.add_subplot(gs[0,2])
    keys = ["Precision","Recall","F1","1−FPR"]
    vals = [prec,rec,f1s,[1-f for f in fprs]]
    x=np.arange(len(keys)); w=0.18
    for i,(sc,col) in enumerate(zip(scenarios,COLORS)):
        ax2.bar(x+i*w,[vals[j][i] for j in range(4)],w,
                label=sc.value.split(" ")[0],color=col,edgecolor="black",linewidth=0.5)
    ax2.set_xticks(x+w*1.5); ax2.set_xticklabels(keys,fontsize=8)
    ax2.set_ylim(0,1.15); ax2.set_title("Metriche Aggregate",fontweight="bold")
    ax2.legend(fontsize=7)
    ax3 = fig.add_subplot(gs[1,:])
    for (sc,m),col in zip(results.items(),COLORS):
        if m.threat_scores:
            ax3.hist(m.threat_scores,bins=40,alpha=0.50,color=col,
                     label=sc.value,density=True,edgecolor="black",linewidth=0.3)
    ax3.axvline(ENGAGEMENT_THRESHOLD,color="red",linestyle="--",linewidth=2,
                label=f"Engage ({ENGAGEMENT_THRESHOLD})")
    ax3.axvline(ALERT_THRESHOLD,color="orange",linestyle="--",linewidth=1.5,
                label=f"Alert ({ALERT_THRESHOLD})")
    ax3.set_xlabel("Threat Score",fontsize=11); ax3.set_ylabel("Densità",fontsize=11)
    ax3.set_title("Distribuzione Threat Score",fontweight="bold"); ax3.legend(fontsize=9)
    plt.savefig("laws_sim_v3_results.png",dpi=150,bbox_inches="tight")
    console.print("[green]✓ Plot salvato → laws_sim_v3_results.png[/green]")


# ══════════════════════════════════════════════════════════════════════
# DEMO MODE — standalone adversarial patch
# ══════════════════════════════════════════════════════════════════════

def run_demo_patch(image_path: str, model_path: str = "yolov8n.pt",
                   steps: int = PATCH_STEPS):
    """
    Demo adversarial patch EoT standalone.

    Uso:
        python laws_sim_v3.py --demo-patch mia_foto.jpg
        python laws_sim_v3.py --demo-patch webcam

    Flusso:
    1. Carica immagine (o webcam)
    2. Rilevamento YOLO baseline → confidence person
    3. Ottimizzazione patch EoT (n_steps iterazioni)
    4. Applica patch → nuovo rilevamento YOLO
    5. Stampa confidence before/after + plot risultati
    6. Calcola CLAE con costo pixel reale
    """
    if not HAS_TORCH or not HAS_YOLO:
        console.print("[red]Demo richiede: pip install torch ultralytics[/red]")
        return

    console.print(Panel(
        f"[bold cyan]Demo Adversarial Patch — EoT[/bold cyan]\n"
        f"Immagine: [yellow]{image_path}[/yellow]\n"
        f"Modello: [yellow]{model_path}[/yellow]\n"
        f"Step ottimizzazione: [yellow]{steps}[/yellow]\n"
        f"EoT trasformazioni/step: [yellow]{EOT_N_TRANSFORMS}[/yellow]",
        border_style="cyan"
    ))

    optimizer = PatchOptimizer(model_path=model_path)
    result    = optimizer.optimize(image_path, n_steps=steps, verbose=True)

    # CLAE con costo pixel reale
    cov      = result["patch_coverage"]
    delta_f1_proxy = result["conf_drop"]   # proxy: drop confidence anziché F1
    clae_real = delta_f1_proxy / cov if cov > 0 else 0.0

    console.print(Panel(
        f"[bold]Riepilogo attacco[/bold]\n\n"
        f"Confidence prima: [green]{result['conf_before']:.4f}[/green]\n"
        f"Confidence dopo:  [red]{result['conf_after']:.4f}[/red]\n"
        f"Drop:             [bold yellow]{result['conf_drop']:+.4f} "
        f"({result['conf_drop']/max(result['conf_before'],1e-6)*100:.1f}%)[/bold yellow]\n\n"
        f"[bold]CLAE con costo pixel reale[/bold]\n"
        f"C_vision (pixel ratio) = {cov:.4f}  ({cov*100:.1f}% del bbox)\n"
        f"CLAE = Δconf / C_vision = {result['conf_drop']:.4f} / {cov:.4f} "
        f"= [bold cyan]{clae_real:.3f}[/bold cyan]",
        title="[bold green]Risultato[/bold green]", border_style="green"
    ))

    PatchOptimizer.save_results(result, output_dir=".")

    # Salva patch tensor per uso nel simulatore
    torch.save(result["patch"], "care_kit_patch.pt")
    console.print("[green]✓ Patch salvata → care_kit_patch.pt[/green]")
    console.print("[dim]Usa con: python laws_sim_v3.py --real-yolo "
                  "--patch care_kit_patch.pt[/dim]")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LAWS-SIM v3.0")
    parser.add_argument("--steps",       type=int,  default=150)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--real-yolo",   action="store_true",
                        help="Usa YOLOv8 reale (richiede yolov8n.pt)")
    parser.add_argument("--image-dir",   type=str, default=None,
                        help="Directory immagini VisDrone per real mode")
    parser.add_argument("--demo-patch",  type=str, default=None,
                        metavar="IMAGE",
                        help="Demo adversarial patch su immagine (o 'webcam')")
    parser.add_argument("--patch",       type=str, default=None,
                        help="Carica patch .pt pre-ottimizzata per simulazione")
    parser.add_argument("--patch-steps", type=int, default=PATCH_STEPS,
                        help=f"Step ottimizzazione patch demo (default {PATCH_STEPS})")
    args = parser.parse_args()

    # ── Demo mode ─────────────────────────────────────────────────────
    if args.demo_patch:
        run_demo_patch(args.demo_patch, steps=args.patch_steps)
        return

    # ── Carica patch pre-ottimizzata ──────────────────────────────────
    patch_tensor = None
    if args.patch and HAS_TORCH:
        patch_tensor = torch.load(args.patch)
        console.print(f"[green]✓ Patch caricata: {args.patch} "
                      f"({patch_tensor.shape})[/green]")

    if HAS_RICH:
        console.print(Panel(
            f"[bold cyan]LAWS-SIM v3.0[/bold cyan]\n"
            f"Steps={args.steps}  Seed={args.seed}  "
            f"RealYOLO={'✓' if args.real_yolo else '✗'}  "
            f"Patch={'✓' if patch_tensor is not None else '✗'}",
            border_style="cyan"))

    # ── Simulazione 4 scenari ─────────────────────────────────────────
    results  = {}
    scenarios = list(AttackScenario)

    for sc in scenarios:
        label = f"Simulazione: {sc.value}…"
        if HAS_RICH:
            with console.status(label):
                sim = LAWSSim(sc, args.steps, args.seed,
                              real_mode=args.real_yolo,
                              image_dir=args.image_dir,
                              patch_tensor=patch_tensor)
                results[sc] = sim.run(verbose=args.verbose)
        else:
            print(label, end=" ", flush=True)
            sim = LAWSSim(sc, args.steps, args.seed,
                          real_mode=args.real_yolo,
                          image_dir=args.image_dir,
                          patch_tensor=patch_tensor)
            results[sc] = sim.run()
            print(f"F1={results[sc].f1:.3f}")

    baseline = results[AttackScenario.NONE]
    print_results(results, baseline)

    # Export JSON
    costs = clae_costs()
    export = {}
    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        export[sc.value] = {
            "precision": round(m.precision,4), "recall": round(m.recall,4),
            "f1": round(m.f1,4), "fpr": round(m.fpr,4),
            "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "clae_v2": round(clae,4) if clae else None,
            "cost_measurable": round(costs.get(sc.name) or 0, 4),
        }
    with open("laws_sim_v3_results.json","w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    console.print("[green]✓ Export → laws_sim_v3_results.json[/green]")

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()