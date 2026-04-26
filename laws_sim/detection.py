"""
Agente Vision con YOLO per rilevamento di persone e oggetti, con supporto per modalità reale (su immagini VisDrone o webcam) e analitica (modello basato su distanza e patch).

paper:
- Sodhro, A.H. et al. (2025): YOLOv8 outdoor confidence 99.1%
  https://doi.org/10.1016/j.iot.2025.101707
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from pathlib import Path

from PIL import Image

from config import (
    YOLO_MAX_RANGE, BASELINE_CONFIDENCE, DETECTION_THRESHOLD,
    PATCH_SUPPRESSION, PATCH_DIST_FALLOFF, PATCH_BBOX_COVERAGE,
    IMG_SIZE, PERSON_CLASS_ID
)
from entities import SimEntity, AgentRole

# Gestione errori importazione YOLO
try:
    import torch
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Gestione errori importazione OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class VisionDetection:
    """Output del modulo vision"""
    detected: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_label: str
    patch_active: bool = False
    patch_coverage: float = 0.0
    real_yolo: bool = False


class VisionAgentReal:
    """
    Agente Vision con supporto YOLOv8 reale

    In modalità reale, utilizza YOLOv8 su immagini effettive (VisDrone / webcam)
    In modalità analitica, utilizza un modello basato sulla fisica
    """

    DETECTION_THRESHOLD = DETECTION_THRESHOLD

    def __init__(self, real_mode: bool = False, model_path: str = "yolov8n.pt",
                 image_dir: Optional[str] = None, patch_tensor: Optional["torch.Tensor"] = None):
        self.real_mode = real_mode and HAS_YOLO
        self.model_path = model_path
        self.image_dir = image_dir
        self.patch_tensor = patch_tensor
        self._model = None
        self._frames: List[Path] = []
        self._frame_idx = 0
        self.fn_count = 0
        self.det_count = 0

        if self.real_mode:
            self._load_model()
            if image_dir:
                self._index_frames()
    #carico YOLO 
    def _load_model(self):
        try:
            self._model = _YOLO(self.model_path)
            for p in self._model.model.parameters():
                p.requires_grad_(False)
            self._model.model.eval()
        except Exception as e:
            print(f"Error loading YOLO: {e} → falling back to analytical model")
            self.real_mode = False

    def _index_frames(self):
        exts = {".jpg", ".jpeg", ".png"}
        self._frames = sorted(
            [f for f in Path(self.image_dir).rglob("*") if f.suffix.lower() in exts]
        )

    def _next_frame(self) -> Optional[Image.Image]:
        if not self._frames:
            return None
        f = self._frames[self._frame_idx % len(self._frames)]
        self._frame_idx += 1
        return Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    def detect(self, entity: SimEntity, distance: float, patch_active: bool = False) -> VisionDetection:
        """Rivela una entità con YOLO"""
        self.det_count += 1
        if self.real_mode and self._frames:
            return self._detect_real_from_sim(entity, distance, patch_active)
        return self._detect_analytical(entity, distance, patch_active)

    def _detect_analytical(self, entity: SimEntity, distance: float, patch_active: bool) -> VisionDetection:
        """Modello analitico basato sulla fisica (fallback)."""
        norm = distance / YOLO_MAX_RANGE
        base = float(np.clip(
            BASELINE_CONFIDENCE * math.exp(-1.5 * norm) + random.gauss(0, 0.04),
            0, 1
        ))

        if patch_active and entity.care_kit_active:
            s_eff = PATCH_SUPPRESSION * math.exp(-PATCH_DIST_FALLOFF * distance)
            conf = float(np.clip(base * (1.0 - s_eff + random.gauss(0, 0.08)), 0, 1))
            cov = PATCH_BBOX_COVERAGE
        else:
            conf = base
            cov = 0.0

        detected = conf >= self.DETECTION_THRESHOLD
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected=detected,
            confidence=conf,
            bbox=(entity.x, entity.y, 5, 8),
            class_label="person" if detected else "background",
            patch_active=patch_active and entity.care_kit_active,
            patch_coverage=cov,
            real_yolo=False
        )

    def _detect_real_from_sim(self, entity: SimEntity, distance: float, patch_active: bool) -> VisionDetection:
        """Rilevamento reale su immagini simulate con supporto patch"""
        frame = self._next_frame()
        if frame is None:
            return self._detect_analytical(entity, distance, patch_active)

        img_t = torch.from_numpy(
            np.array(frame).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        apply_p = patch_active and entity.care_kit_active and self.patch_tensor is not None

        if apply_p:
            # Applico la patch al centro dell'immagine (simulando l'effetto sulla persona) -> in un caso reale, la patch sarebbe posizionata in base alla posizione stimata della persona
            from patch_optimizer import PatchOptimizer
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            bbox_px = (cx - 50, cy - 80, cx + 50, cy + 80)
            img_t = PatchOptimizer.apply_patch_to_image(img_t, self.patch_tensor, bbox_px)
            cov = (self.patch_tensor.shape[1] * self.patch_tensor.shape[2]) / max((100 * 160), 1)
        else:
            cov = 0.0

        img_pil = Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        results = self._model(img_pil, verbose=False)

        yolo_conf = self._get_person_conf(results)

        dist_scale = math.exp(-1.0 * distance / YOLO_MAX_RANGE)
        conf = float(np.clip(yolo_conf * dist_scale + random.gauss(0, 0.02), 0, 1))

        detected = conf >= self.DETECTION_THRESHOLD
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected=detected,
            confidence=conf,
            bbox=(entity.x, entity.y, 5, 8),
            class_label="person" if detected else "background",
            patch_active=apply_p,
            patch_coverage=cov,
            real_yolo=True
        )

    @staticmethod
    def _get_person_conf(results) -> float:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0