"""
Adversarial patch optimizer with Expectation over Transformation (EoT).

Reference:
- Athalye, A., Sutskever, I. (2017). "Synthesizing Robust Adversarial Examples".
  https://arxiv.org/abs/1707.07397
"""

import random
import numpy as np
from typing import Optional, Tuple

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


def get_chest_bbox(person_bbox, patch_w, patch_h, img_size=640):
    """Restituisce un bbox per la patch centrato sul petto."""
    x1, y1, x2, y2 = person_bbox
    cx = (x1 + x2) // 2
    cy = y1 + int((y2 - y1) * 0.33)
    px1 = cx - patch_w // 2
    py1 = cy - patch_h // 2
    px2 = cx + patch_w // 2
    py2 = cy + patch_h // 2
    px1 = max(0, px1); py1 = max(0, py1)
    px2 = min(img_size, px2); py2 = min(img_size, py2)
    return (px1, py1, px2, py2)


class PatchOptimizer:
    def __init__(self, patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 model_path: str = "yolov8n.pt"):
        if not HAS_YOLO:
            raise RuntimeError("Ultralytics YOLO not available")
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.model_path = model_path
        self._yolo = None
        self.patch = torch.rand(3, patch_h, patch_w, dtype=torch.float32, requires_grad=True)

    def _get_model(self):
        if self._yolo is None:
            self._yolo = _YOLO(self.model_path)
            for p in self._yolo.model.parameters():
                p.requires_grad_(False)
            self._yolo.model.eval()
        return self._yolo

    @staticmethod
    def _random_transform(patch: torch.Tensor) -> torch.Tensor:
        t = patch.unsqueeze(0)
        angle = random.uniform(-20, 20)
        t = TF.rotate(t, angle)
        scale = random.uniform(0.75, 1.25)
        new_h = max(1, int(patch.shape[1] * scale))
        new_w = max(1, int(patch.shape[2] * scale))
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        t = F.interpolate(t, size=(patch.shape[1], patch.shape[2]), mode="bilinear", align_corners=False)
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.8, 1.2)
        t = TF.adjust_brightness(t, brightness)
        t = TF.adjust_contrast(t, contrast)
        if random.random() < 0.5:
            t = TF.hflip(t)
        sigma = random.uniform(0, 1.5)
        if sigma > 0.3:
            k = int(sigma * 4) | 1
            k = max(3, k)
            t = TF.gaussian_blur(t, kernel_size=k, sigma=sigma)
        return torch.clamp(t.squeeze(0), 0, 1)

    @staticmethod
    def apply_patch_to_image(img_t: torch.Tensor, patch_t: torch.Tensor,
                             bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        ph = y2 - y1
        pw = x2 - x1
        if ph <= 0 or pw <= 0:
            return img_t
        patch_resized = F.interpolate(patch_t.unsqueeze(0), size=(ph, pw),
                                      mode="bilinear", align_corners=False).squeeze(0)
        img_clone = img_t.clone()
        img_clone[:, y1:y2, x1:x2] = patch_resized
        return img_clone

    @staticmethod
    def _get_person_conf(results) -> float:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0

    @staticmethod
    def _get_person_conf_tensor(results) -> torch.Tensor:
        """Restituisce la confidence media come tensore differenziabile."""
        conf_list = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    conf_list.append(r.boxes.conf[mask])
        if conf_list:
            return torch.cat(conf_list).mean()
        # Importante: restituire un tensore che NON richiede gradiente esplicitamente,
        # altrimenti si stacca dal grafo. Usiamo un tensore zero normale.
        return torch.tensor(0.0, device=results[0].boxes.conf.device if results else 'cpu')

    def optimize(self, image_path: str,
                 bbox: Optional[Tuple[int, int, int, int]] = None,
                 n_steps: int = PATCH_STEPS,
                 lr: float = PATCH_LR,
                 n_eot: int = EOT_N_TRANSFORMS,
                 verbose: bool = True) -> dict:
        model = self._get_model()

        # Carica immagine
        if image_path == "webcam":
            if not HAS_CV2:
                raise RuntimeError("OpenCV required for webcam")
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            for _ in range(10):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError("Webcam read failed")
            cv2.imwrite("webcam_debug_optimize.jpg", frame)
            print("✓ Frame salvato in webcam_debug_optimize.jpg")
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.open(image_path).convert("RGB")

        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)

        # Baseline detection
        print("[cyan]Rilevamento baseline (senza patch)...[/cyan]")
        results_base = model(img_pil, verbose=False)
        conf_before = self._get_person_conf(results_base)
        print(f"  Confidence person PRIMA patch: [bold]{conf_before:.4f}[/bold]")

        # Determina bbox persona
        person_bbox = bbox if bbox is not None else self._find_person_bbox(results_base)
        if person_bbox is None:
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            person_bbox = (cx - 50, cy - 80, cx + 50, cy + 80)

        # Bbox per la patch sul petto
        patch_bbox = get_chest_bbox(person_bbox, self.patch_w, self.patch_h, IMG_SIZE)
        x1, y1, x2, y2 = patch_bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        patch_area = self.patch_h * self.patch_w
        coverage = patch_area / max(bbox_area, 1)

        print(f"  BBox persona: {person_bbox}")
        print(f"  BBox patch (petto): {patch_bbox} (area={bbox_area}px)")
        print(f"  Patch: {self.patch_h}×{self.patch_w}  coverage={coverage:.3f} ({coverage*100:.1f}% del bbox patch)")

              # Black‑box optimization: Adaptive Random Search with EoT
        best_patch = self.patch.clone()
        best_loss = float('inf')
        loss_history = []
        sigma = 0.1               # perturbazione iniziale
        patience = 15             # step senza miglioramenti prima di ridurre sigma
        no_improve = 0

        def compute_eot_loss(patch_tensor):
            """Calcola la loss media EoT (confidence persona) per un dato tensore patch."""
            losses = []
            for _ in range(n_eot):
                patch_t = self._random_transform(patch_tensor)
                img_patched = self.apply_patch_to_image(img_t, patch_t, patch_bbox)
                batch = img_patched.unsqueeze(0)
                results = model(batch, verbose=False)
                conf = self._get_person_conf(results)   # float
                losses.append(conf)
            return np.mean(losses)

        if verbose:
            print(f"\n[cyan]Black‑box Adaptive Random Search: {n_steps} steps, {n_eot} transforms/step...[/cyan]")

        for step in range(n_steps):
            # Genera candidato con rumore gaussiano
            noise = torch.randn_like(best_patch) * sigma
            candidate = torch.clamp(best_patch + noise, 0, 1)
            candidate_loss = compute_eot_loss(candidate)

            # Accettazione probabilistica (simulated annealing)
            if candidate_loss < best_loss:
                accept = True
            else:
                # Probabilità di accettare peggioramenti (decresce col tempo)
                temperature = max(0.01, 0.5 * (1 - step / n_steps))
                delta = candidate_loss - best_loss
                p_accept = np.exp(-delta / temperature)
                accept = random.random() < p_accept

            if accept:
                best_patch = candidate.clone()
                best_loss = candidate_loss
                self.patch.data.copy_(best_patch)
                no_improve = 0
            else:
                no_improve += 1

            # Decadimento adattivo di sigma
            if no_improve >= patience:
                sigma *= 0.8
                no_improve = 0
                if verbose:
                    print(f"  Step {step+1:3d}: riduzione sigma a {sigma:.4f}")

            loss_history.append(best_loss)

            if verbose and (step % 10 == 0 or step == n_steps - 1):
                print(f"  Step {step+1:3d}/{n_steps} best_loss={best_loss:.4f} sigma={sigma:.4f}")

        # Alla fine, imposta la patch migliore trovata
        self.patch.data.copy_(best_patch)

        # Valutazione finale
        print("\n[cyan]Evaluation with optimized patch...[/cyan]")
        img_final = self.apply_patch_to_image(img_t, self.patch.detach(), patch_bbox)
        img_final_pil = Image.fromarray((img_final.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        results_after = model(img_final_pil, verbose=False)
        conf_after = self._get_person_conf(results_after)

        print(f"  Person confidence AFTER patch: [bold red]{conf_after:.4f}[/bold red]")
        drop = conf_before - conf_after
        print(f"  Drop: [bold yellow]{drop:+.4f} ({drop/max(conf_before,1e-6)*100:.1f}%)[/bold yellow]")

        return {
            "patch": self.patch.detach(),
            "conf_before": conf_before,
            "conf_after": conf_after,
            "conf_drop": drop,
            "loss_history": loss_history,
            "patch_coverage": coverage,
            "bbox": patch_bbox,
            "img_original": img_t,
            "img_patched": img_final,
        }

    @staticmethod
    def _find_person_bbox(results) -> Optional[Tuple[int, int, int, int]]:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    box = r.boxes.xyxy[mask][0].int().tolist()
                    return tuple(box)
        return None