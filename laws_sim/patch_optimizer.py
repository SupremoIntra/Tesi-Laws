"""
Adversarial Patch Optimizer — Expectation over Transformation (EoT).

Riferimenti:
    [1] Athalye et al. (2017). "Synthesizing Robust Adversarial Examples".
        https://arxiv.org/abs/1707.07397
    [2] Thys et al. (2019). "Fooling Automated Surveillance Cameras".
        https://arxiv.org/abs/1904.08653
    [3] Brown et al. (2017). "Adversarial Patch".
        https://arxiv.org/abs/1712.09665
"""

import random
from typing import Optional, Tuple, List

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



# UTILITY: POSIZIONAMENTO PATCH SUL PETTO (da migliorare!)

def get_chest_bbox(person_bbox: Tuple[int,int,int,int],
                   patch_w: int, patch_h: int,
                   img_size: int = 640) -> Tuple[int,int,int,int]:
    """Centra la patch sul petto -> simulato 33% dall'alto del bbox persona, non al centro esatto"""
    x1, y1, x2, y2 = person_bbox
    cx = (x1 + x2) // 2
    cy = y1 + int((y2 - y1) * 0.33)
    px1 = max(0,        cx - patch_w // 2)
    py1 = max(0,        cy - patch_h // 2)
    px2 = min(img_size, cx + patch_w // 2)
    py2 = min(img_size, cy + patch_h // 2)
    return (px1, py1, px2, py2)



# ANCHOR MASK — individua le celle di YOLO che fanno overlap sul bbox della patch -> mi focalizzo solo su quelle per aumentare il gradiente e forzare YOLO a vedere la PATCH come unica fonte di informazione per la detection -> non usa più ancore di backup fuori dalla patch (sul corpo o ambiente..)

def build_anchor_mask(patch_bbox: Tuple[int,int,int,int],
                      img_size: int = 640,
                      device: str = "cpu") -> Optional[torch.Tensor]:
    """Calcola gli indici delle celle YOLOv8 che si sovrappongono al patch_bbox."""
    x1, y1, x2, y2 = patch_bbox
    indices = []
    offset = 0

    # YOLOv8 ha 3 scale di output (strides 8, 16, 32) → 80x80, 40x40, 20x20 griglie 
    # Ogni cella ha un centro (cx, cy) che rappresenta l'ancora. Se il centro cade dentro il patch_bbox (con un margine), la cella è attiva.
    for stride in [8, 16, 32]:
        grid   = img_size // stride
        margin = stride * 0.5
        for gy in range(grid):
            for gx in range(grid):
                cx = (gx + 0.5) * stride
                cy = (gy + 0.5) * stride
                if (x1 - margin <= cx <= x2 + margin and
                        y1 - margin <= cy <= y2 + margin):
                    indices.append(offset + gy * grid + gx)
        offset += grid * grid

    if not indices:
        print(f"  ATTENZIONE: nessuna cella YOLO nel patch_bbox {patch_bbox}")
        return None

    n = len(indices)
    print(f"  Celle YOLO nel patch bbox: {n}/8400  "
          f"(amplificazione gradiente: ×{8400/n:.0f})")
    return torch.tensor(indices, dtype=torch.long, device=device)



# PATCH OPTIMIZER — ottimizza la patch con EoT, focalizzandosi solo sulle celle attive (anchor mask) per massimizzare l'efficacia dell'attacco e forzare YOLO a dipendere esclusivamente dalla patch per la detection.
class PatchOptimizer:
    TV_WEIGHT = 0.001 # peso della total variation (regolarizzazione per mantenere la patch visivamente coerente)

    def __init__(self, patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 model_path: str = "yolov8n.pt"):
        if not HAS_YOLO:
            raise RuntimeError("pip install ultralytics")
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.model_path = model_path
        self._yolo      = None
        self.patch = (torch.rand(3, patch_h, patch_w) * 0.4 + 0.3).requires_grad_(True)

    def _get_model(self):
        """Carica YOLOv8 lazy (solo alla prima chiamata)"""
        if self._yolo is None:
            self._yolo = _YOLO(self.model_path)
            for p in self._yolo.model.parameters():
                p.requires_grad_(False) # congeliamo i pesi di YOLO, ottimizziamo solo la patch
            self._yolo.model.eval()

            # FIX: AUTOGRAD GRAPH BREAK -> evitiamo che YOLO generi dinamicamente i tensori 'anchors' e 'strides' fuori dal grafo computazionale
            # YOLOv8 genera 'anchors' e 'strides' dinamicamente al primo forward.
            # Eseguendo un forward con un tensore dummy (fuori da torch.no_grad()),
            # forziamo PyTorch a inizializzare questi tensori agganciandoli 
            # correttamente al grafo computazionale. Senza questo fix, 
            # la backpropagation verso i pixel della patch fallisce silenziosamente.
            device = next(self._yolo.model.parameters()).device
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
            _ = self._yolo.model(dummy)

        return self._yolo

    @staticmethod
    def _focused_bce_loss(torch_model: torch.nn.Module,
                          batch: torch.Tensor,
                          person_class: int = PERSON_CLASS_ID) -> torch.Tensor:
        """abbatte la confidence di detection, attaccando tutte le ancore attive."""
        raw  = torch_model(batch)
        pred = raw[0] if isinstance(raw, (tuple, list)) else raw
        
        person_scores = pred[:, 4 + person_class, :] 
        
        # Selezioniamo TUTTE le celle che "vedono" una persona -> CONFIDENCE > 0.10 (AGGRESSIVO)
        # Attacca qualsiasi cella di YOLO che pensi, anche solo al 10%, che ci sia una persona -> questo forza YOLO a dipendere esclusivamente dalla patch per la detection, senza usare ancore di backup su altre parti del corpo o ambiente.
        
        mask = person_scores > 0.10
        
        if not mask.any():
            # YOLO non vede più niente, l'attacco ha vinto!
            # Restituiamo una loss dummy a 0 per non far crashare l'ottimizzatore
            return torch.tensor(0.0, device=batch.device, requires_grad=True)

        active_scores = person_scores[mask]
        return F.binary_cross_entropy(active_scores, torch.zeros_like(active_scores))

    @staticmethod
    def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
        """Total Variation: penalizza variazioni brusche tra pixel vicini per mantenere la patch visivamente coerente."""
        dx = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
        dy = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
        return dx + dy

    @staticmethod
    def _random_transform(patch: torch.Tensor) -> torch.Tensor:
        """Trasformazione fisica casuale (EoT): rotazione, scala, luminosità, contrasto, blur"""
        t = patch.unsqueeze(0)
        t = TF.rotate(t, random.uniform(-20, 20))

        scale = random.uniform(0.75, 1.25)
        nh    = max(1, int(patch.shape[1] * scale))
        nw    = max(1, int(patch.shape[2] * scale))
        t = F.interpolate(t, (nh, nw), mode="bilinear", align_corners=False)
        t = F.interpolate(t, (patch.shape[1], patch.shape[2]),
                          mode="bilinear", align_corners=False)

        t = TF.adjust_brightness(t, random.uniform(0.6, 1.4))
        t = TF.adjust_contrast(t,   random.uniform(0.7, 1.3))

    

        sigma = random.uniform(0, 1.0)
        if sigma > 0.3:
            t = TF.gaussian_blur(t, kernel_size=max(3, int(sigma*4)|1), sigma=sigma)

        return torch.clamp(t.squeeze(0), 0, 1)

    @staticmethod
    def apply_patch_to_image(img_t: torch.Tensor,
                             patch_t: torch.Tensor,
                             bbox: Tuple[int,int,int,int]) -> torch.Tensor:
        """tensor slicing -> sovrappongo patch all'img originale nel bbox"""
        x1, y1, x2, y2 = bbox
        ph, pw = y2-y1, x2-x1
        if ph <= 0 or pw <= 0:
            return img_t
        p = F.interpolate(patch_t.unsqueeze(0), (ph,pw),
                          mode="bilinear", align_corners=False).squeeze(0)
        out = img_t.clone()
        out[:, y1:y2, x1:x2] = p
        return out

    @staticmethod
    def _get_person_conf(results) -> float:
        """Confidence YOLO (post-NMS)"""
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0

    @staticmethod
    def _find_person_bbox(results) -> Optional[Tuple[int,int,int,int]]:
        """Bbox della persona con confidence più alta"""
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return tuple(r.boxes.xyxy[mask][0].int().tolist())
        return None

    def optimize(self, image_path: str,
                 bbox: Optional[Tuple[int,int,int,int]] = None,
                 n_steps: int = PATCH_STEPS,
                 lr: float    = PATCH_LR,
                 n_eot: int   = EOT_N_TRANSFORMS,
                 verbose: bool = True) -> dict:
        
        model = self._get_model()

        #Carica img (o webcam) 
        if image_path == "webcam":
            if not HAS_CV2:
                raise RuntimeError("pip install opencv-python")
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError("Webcam non disponibile")
            cv2.imwrite("webcam_capture.jpg", frame)
            print("Frame salvato → webcam_capture.jpg")
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.open(image_path).convert("RGB")

        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_t   = torch.from_numpy(
            np.array(img_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        #Baseline
        print("Rilevamento baseline...")
        results_base = model(img_pil, verbose=False)
        conf_before  = self._get_person_conf(results_base)
        print(f"  Confidence PRIMA patch: {conf_before:.4f}")
        if conf_before < 0.10:
            print("  Attenzione: confidence bassa. Persona non rilevata correttamente -> cambia img o sistema l'ambiente")

        #Calcola bbox petto
        person_bbox = bbox if bbox is not None else self._find_person_bbox(results_base)
        if person_bbox is None:
            cx, cy  = IMG_SIZE // 2, IMG_SIZE // 2
            person_bbox = (cx-80, cy-120, cx+80, cy+120)
            print("  Persona non trovata → bbox centrale")

        #FIX: stavo dividendo l'area della patch per l'area della patch...
        patch_bbox = get_chest_bbox(person_bbox, self.patch_w, self.patch_h, IMG_SIZE)
        px1, py1, px2, py2 = person_bbox  # Prendo i vertici della PERSONA
        person_area = (px2 - px1) * (py2 - py1)
        coverage = (self.patch_h * self.patch_w) / max(person_area, 1) #... area patch / area persona
        print(f"  BBox persona: {person_bbox}")
        print(f"  BBox patch (petto): {patch_bbox}  coverage={coverage:.3f}")

        #Setup
        torch_model = model.model
        device      = next(torch_model.parameters()).device
        img_t       = img_t.to(device)

        self.patch  = self.patch.detach().to(device).requires_grad_(True)
        optimizer   = torch.optim.Adam([self.patch], lr=lr) #adam è più stabile per ottimizzare pixel direttamente (in futuro provo anche SGD con momentum)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.1)

        loss_history = []
        conf_history = []

        if verbose:
            print(f"\nOttimizzazione: {n_steps} step × {n_eot} EoT  |  "
                  f"LR cosine {lr}→{lr*0.1:.4f}  |  TV_weight={self.TV_WEIGHT}")
            print("─" * 65)

        for step in range(n_steps):
            optimizer.zero_grad()
            step_losses = []
        # Per ogni step, applichiamo n_eot trasformazioni casuali alla patch e calcoliamo la loss media → questo rende la patch robusta a variazioni fisiche e di illuminazione, simulando condizioni reali di utilizzo.
            for _ in range(n_eot):
                patch_t     = self._random_transform(self.patch).to(device)
                img_patched = self.apply_patch_to_image(img_t, patch_t, patch_bbox)
                adv_loss    = self._focused_bce_loss(
                    torch_model, img_patched.unsqueeze(0))
                step_losses.append(adv_loss)

            adv  = torch.stack(step_losses).mean()
            tv   = self._tv_loss(self.patch)
            loss = adv + self.TV_WEIGHT * tv
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                self.patch.clamp_(0, 1)

            loss_history.append(adv.item())

            if verbose and (step % 10 == 0 or step == n_steps - 1):
                with torch.no_grad():
                    img_chk = self.apply_patch_to_image(img_t, self.patch, patch_bbox)
                    pil_chk = Image.fromarray(
                        (img_chk.cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
                    conf_now = self._get_person_conf(model(pil_chk, verbose=False))
                    conf_history.append((step, conf_now))
                print(f"  step {step+1:3d}/{n_steps}  "
                      f"bce={adv.item():.4f}  tv={tv.item():.4f}  "
                      f"conf_yolo={conf_now:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.5f}")

        #Valutazione finale
        print("\nValutazione finale...")
        with torch.no_grad():
            img_final = self.apply_patch_to_image(img_t, self.patch, patch_bbox)
        pil_final     = Image.fromarray(
            (img_final.cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
        conf_after    = self._get_person_conf(model(pil_final, verbose=False))
        drop          = conf_before - conf_after

        print(f"  Confidence PRIMA:  {conf_before:.4f}")
        print(f"  Confidence DOPO:   {conf_after:.4f}")
        print(f"  Drop:              {drop:+.4f}  ({drop/max(conf_before,1e-6)*100:.1f}%)")
        if coverage > 0:
            print(f"  CEAE = {drop:.4f} / {coverage:.4f} = {drop/coverage:.3f}")

        return {
            "patch"         : self.patch.detach().cpu(),
            "conf_before"   : conf_before,
            "conf_after"    : conf_after,
            "conf_drop"     : drop,
            "loss_history"  : loss_history,
            "conf_history"  : conf_history,
            "patch_coverage": coverage,
            "bbox"          : patch_bbox,
            "img_original"  : img_t.cpu(),
            "img_patched"   : img_final.cpu(),
        }