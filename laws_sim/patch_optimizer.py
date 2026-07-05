"""
Adversarial Patch Optimizer — Targeted Saliency Attack, Full EoT & O(1) CPU Forward.

Riferimenti Accademici:
    [1] Carlini & Wagner (2017). "Towards Evaluating the Robustness of Neural Networks".
    [2] Wu et al. (2020). "Making an Invisibility Cloak: Real World Adversarial Attacks 
        on Object Detectors". 
    [3] Athalye et al. (2017). "Synthesizing Robust Adversarial Examples".
    [4] Ottimizzazione Tensoriale Avanzata: Parametrizzazione Sigmoide e Area-based Sampling.
"""

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from config import (
    PATCH_H, PATCH_W, PATCH_LR, PATCH_STEPS,
    EOT_N_TRANSFORMS, PERSON_CLASS_ID, IMG_SIZE
)

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

def get_chest_bbox_proportional(person_bbox: Tuple[int,int,int,int],
                                img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    """Posizionamento proporzionale (40% width, 30% height)"""
    x1, y1, x2, y2 = person_bbox
    w, h = x2 - x1, y2 - y1
    pw = max(int(w * 0.40), 4)
    ph = max(int(h * 0.30), 4)
    cx = x1 + w // 2
    cy = y1 + int(h * 0.33)
    px1 = max(0, cx - pw // 2)
    py1 = max(0, cy - ph // 2)
    px2 = min(img_w, px1 + pw)
    py2 = min(img_h, py1 + ph)
    return (px1, py1, px2, py2)

class PatchOptimizer:
    TV_WEIGHT = 0.001 
    TARGET_CONF = 0.35 

    def __init__(self, patch_h: int = PATCH_H, patch_w: int = PATCH_W,
                 model_path: str = "yolov8n.pt"):
        if not HAS_YOLO:
            raise RuntimeError("pip install ultralytics")
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.model_path = model_path
        self._yolo      = None
        
        # Parametrizzazione Sigmoide (Evita dead pixels e gradient vanishing ai bordi)
        self.patch_logits = (torch.randn(3, patch_h, patch_w, device="cpu") * 0.1).requires_grad_(True)

    def _get_model(self):
        if self._yolo is None:
            self._yolo = _YOLO(self.model_path)
            self._yolo.to("cpu")  # CPU Enforced
            for p in self._yolo.model.parameters():
                p.requires_grad_(False)
            self._yolo.model.eval()

            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device="cpu")
            _ = self._yolo.model(dummy)

        return self._yolo

    @staticmethod
    def _build_spatial_mask(person_bbox: Tuple[int,int,int,int], 
                            img_size: int = 640, 
                            device: str = "cpu") -> torch.Tensor:
        x_min, y_min, x_max, y_max = person_bbox
        strides = [8, 16, 32]
        mask_list = []
        
        for s in strides:
            grid_size = img_size // s
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_size, device=device), 
                torch.arange(grid_size, device=device), 
                indexing='ij'
            )
            cx = (grid_x + 0.5) * s
            cy = (grid_y + 0.5) * s
            in_box = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
            mask_list.append(in_box.flatten())
            
        return torch.cat(mask_list, dim=0)

    @staticmethod
    def _targeted_hinge_loss(torch_model: torch.nn.Module, 
                             batch: torch.Tensor, 
                             spatial_mask: torch.Tensor, 
                             target_conf: float = 0.35,
                             person_class: int = PERSON_CLASS_ID) -> torch.Tensor:
        raw = torch_model(batch)
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw
        
        person_scores = torch.sigmoid(preds[:, 4 + person_class, :]) 
        masked_scores = person_scores[:, spatial_mask] 
        
        hinge = F.relu(masked_scores - target_conf)
        active_hinge = hinge[hinge > 0]
        
        if active_hinge.numel() > 0:
            return active_hinge.mean()
        else:
            return masked_scores.sum() * 0.0

    @staticmethod
    def _visdrone_eot(patch: torch.Tensor, n_eot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = patch.device
        _, h, w = patch.shape
        
        patch_batch = patch.unsqueeze(0).expand(n_eot, -1, -1, -1)
        
        c_factor = torch.empty(n_eot, 1, 1, 1, device=device).uniform_(0.6, 1.4)
        b_factor = torch.empty(n_eot, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
        patch_rgb = patch_batch * c_factor + b_factor
        
        noise = torch.randn_like(patch_rgb) * 0.05
        patch_rgb = torch.clamp(patch_rgb + noise, 0.0, 1.0)
        
        alpha = torch.ones(n_eot, 1, h, w, device=device)
        patch_rgba = torch.cat([patch_rgb, alpha], dim=1)
        
        angles = torch.empty(n_eot, device=device).uniform_(-15.0, 15.0) * (np.pi / 180.0)
        scales = torch.empty(n_eot, device=device).uniform_(0.4, 0.9)
        tx = torch.empty(n_eot, device=device).uniform_(-0.1, 0.1)
        ty = torch.empty(n_eot, device=device).uniform_(-0.1, 0.1)

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        theta = torch.zeros(n_eot, 2, 3, device=device)
        theta[:, 0, 0] = cos_a / scales
        theta[:, 0, 1] = -sin_a / scales
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a / scales
        theta[:, 1, 1] = cos_a / scales
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, patch_rgba.size(), align_corners=False)
        transformed = F.grid_sample(patch_rgba, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        trans_rgb = transformed[:, :3, :, :]
        trans_mask = transformed[:, 3:, :, :]
        
        return trans_rgb, trans_mask

    @staticmethod
    def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
        dx = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
        dy = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
        return dx + dy

    @staticmethod
    def _get_person_conf(results) -> float:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0

    def optimize_universal(self, loader,
                           n_steps: int = PATCH_STEPS,
                           lr: float    = PATCH_LR,
                           n_eot: int   = EOT_N_TRANSFORMS,
                           batch_size: int = 4,
                           verbose: bool = True) -> dict:
        MAX_TARGETS_PER_FRAME = 3   

        model       = self._get_model()
        torch_model = model.model
        device      = "cpu"

        optimizer  = torch.optim.Adam([self.patch_logits], lr=lr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(n_steps), eta_min=lr * 0.1)

        batch_gen    = self._infinite_batches(loader, batch_size)
        loss_history = []
        conf_history = []

        if verbose:
            print(f"\n[Ablation] Saliency Universal Patch | Parametrizzazione Sigmoide, O(1) Canvas")
            print(f"Steps: {n_steps} | EoT: {n_eot} | Target Conf: {self.TARGET_CONF}")
            print("─" * 65)

        for step in range(int(n_steps)):
            imgs_pil, bboxes_list = next(batch_gen)
            
            if not any(bboxes_list):
                continue

            optimizer.zero_grad()
            accumulated_adv_loss = 0.0
            processed_images = 0

            for img_pil, bboxes in zip(imgs_pil, bboxes_list):
                if not bboxes:
                    continue

                img_t = torch.from_numpy(
                    np.array(img_pil).astype(np.float32) / 255.0
                ).permute(2, 0, 1).to(device)

                if len(bboxes) > MAX_TARGETS_PER_FRAME:
                    # Area-based Sampling
                    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:MAX_TARGETS_PER_FRAME]

                global_canvas_rgb = torch.zeros(n_eot, 3, IMG_SIZE, IMG_SIZE, device=device)
                global_canvas_mask = torch.zeros(n_eot, 1, IMG_SIZE, IMG_SIZE, device=device)
                global_spatial_mask_list = []
                targets_in_image = 0

                # Generiamo il tensore corrente DENTRO il ciclo per creare un nuovo 
                # grafo computazionale per ogni immagine, aggirando l'errore autograd.
                current_patch = torch.sigmoid(self.patch_logits)
                trans_rgb, trans_mask = self._visdrone_eot(current_patch, n_eot)

                for person_bbox in bboxes:
                    patch_bbox = get_chest_bbox_proportional(person_bbox, IMG_SIZE, IMG_SIZE)
                    x1, y1, x2, y2 = patch_bbox
                    ph, pw = y2 - y1, x2 - x1
                    if ph <= 0 or pw <= 0: continue

                    spatial_mask = self._build_spatial_mask(person_bbox, IMG_SIZE, device)
                    if not spatial_mask.any(): continue

                    global_spatial_mask_list.append(spatial_mask)

                    trans_rgb_res = F.interpolate(trans_rgb, (ph, pw), mode="bilinear", align_corners=False)
                    trans_mask_res = torch.clamp(F.interpolate(trans_mask, (ph, pw), mode="bilinear", align_corners=False), 0.0, 1.0)

                    pad_left, pad_right = x1, IMG_SIZE - x2
                    pad_top, pad_bottom = y1, IMG_SIZE - y2
                    
                    canvas_rgb = F.pad(trans_rgb_res, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
                    canvas_mask = F.pad(trans_mask_res, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

                    global_canvas_rgb = global_canvas_rgb * (1 - canvas_mask) + canvas_rgb * canvas_mask
                    global_canvas_mask = global_canvas_mask + canvas_mask - (global_canvas_mask * canvas_mask)
                    
                    targets_in_image += 1

                if targets_in_image == 0:
                    continue

                global_spatial_mask = torch.stack(global_spatial_mask_list).any(dim=0)
                adv_batch = img_t.unsqueeze(0).expand(n_eot, -1, -1, -1) * (1 - global_canvas_mask) + global_canvas_rgb * global_canvas_mask

                loss = self._targeted_hinge_loss(torch_model, adv_batch, global_spatial_mask, self.TARGET_CONF)
                loss.backward()
                accumulated_adv_loss += loss.item()
                processed_images += 1

            if processed_images == 0:
                continue

            accumulated_adv_loss /= processed_images

            # Ricreiamo il grafo pulito per la TV Loss
            current_patch_tv = torch.sigmoid(self.patch_logits)
            tv = self._tv_loss(current_patch_tv)
            (self.TV_WEIGHT * tv).backward()
            
            torch.nn.utils.clip_grad_norm_([self.patch_logits], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            loss_history.append(accumulated_adv_loss)

            if verbose and (step % 20 == 0 or step == int(n_steps) - 1):
                with torch.no_grad():
                    img0 = torch.from_numpy(np.array(imgs_pil[0]).astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
                    pb = get_chest_bbox_proportional(bboxes_list[0][0], IMG_SIZE, IMG_SIZE)
                    chk_img = img0.clone()
                    
                    # Generazione patch pura per telemetria visiva (senza logit graph)
                    eval_patch = torch.sigmoid(self.patch_logits).detach()
                    p_res = F.interpolate(eval_patch.unsqueeze(0), (pb[3]-pb[1], pb[2]-pb[0]), mode="bilinear", align_corners=False).squeeze(0)
                    chk_img[:, pb[1]:pb[3], pb[0]:pb[2]] = p_res
                    
                    pil = Image.fromarray((chk_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    conf_now = self._get_person_conf(model(pil, verbose=False))
                    conf_history.append((step, conf_now))
                    
                print(f"  [Step {step:4d}] Hinge L={accumulated_adv_loss:.4f} | TV={tv.item():.4f} | YOLO Conf={conf_now:.4f} | LR={optimizer.param_groups[0]['lr']:.5f}")

        return {
            "patch"         : torch.sigmoid(self.patch_logits).detach().cpu(),
            "loss_history"  : loss_history,
            "conf_history"  : conf_history,
        }

    @staticmethod
    def _infinite_batches(loader, batch_size: int):
        while True:
            yield from loader.iter_batches(batch_size=batch_size, shuffle=True)