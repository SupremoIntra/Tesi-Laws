"""
LAWS-SIM loop del simulator.
"""

import random
import numpy as np
import json
import os

from config import GRID_SIZE
from entities import Environment, AgentRole
from detection import VisionAgentStat
from fusion_decision import FusionAgent, DecisionAgent
from metrics import SimMetrics, AttackScenario


class LAWSSim:
    """Multi-agent LAWS simulator"""

    def __init__(self, scenario: AttackScenario, steps: int, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.scenario = scenario
        self.steps = steps
        self.env = Environment(grid_size=GRID_SIZE)
        
        # IL PONTE: Leggo il dato reale misurato da VisDrone
        vision_f1 = 0.710  # Baseline fittizia di sicurezza
        if os.path.exists("vision_metrics.json"):
            with open("vision_metrics.json", "r") as f:
                data = json.load(f)
                # Se c'è un attacco visivo, carico il crollo dell'F1 (es. 0.008)
                if scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING):
                    vision_f1 = data.get("f1", 0.008)

        # Inizializzo l'Agente Visivo Statistico passandogli il dato empirico
        self.vision = VisionAgentStat(empirical_f1=vision_f1)
        
        self.fusion = FusionAgent()
        self.decision = DecisionAgent()
        self.metrics = SimMetrics()
        self.step_log = []

        # Attivo il care kit (la patch) se lo scenario lo prevede
        if scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING):
            for e in self.env.entities:
                if e.role == AgentRole.TARGET:
                    e.care_kit_active = True

    def run(self, verbose: bool = False) -> SimMetrics:
        patch_on = self.scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING)
        poison_on = self.scenario in (AttackScenario.OSINT_POISONING, AttackScenario.CASCADING)

        for _ in range(self.steps):
            self.env.tick()
            for entity in self.env.visible():
                dist = self.env.dist_to_drone(entity)
                civ_near = self.env.nearby_civilians(entity)

                # OSINT analysis e attacco di Data Poisoning
                osint_p = entity.osint_profile
                
                # Attacco informatico: Altero i metadati nel database!
                if poison_on and entity.role == AgentRole.TARGET:
                    # Nascondo il target abbassando i suoi profili di rischio
                    osint_p.plate_blacklist = False
                    osint_p.geo_risk = max(0.0, osint_p.geo_risk - random.uniform(0.40, 0.60))
                    osint_p.social_matches = 0
                    osint_p.is_poisoned = True
                elif poison_on and entity.role == AgentRole.CIVILIAN:
                    # Colpisco i civili facendoli sembrare terroristi
                    osint_p.plate_blacklist = True
                    osint_p.geo_risk = 0.95
                    osint_p.social_matches = 15
                    osint_p.is_poisoned = True

                vision_d = self.vision.detect(entity, dist, patch_active=patch_on)
                fusion_r = self.fusion.fuse(
                    entity.id, vision_d, osint_p, entity.behavioral_score
                )
                decision = self.decision.decide(entity.id, fusion_r, civ_near)

                # Aggiorno le metriche 
                is_threat = (entity.role == AgentRole.TARGET)
                is_engaged = decision.action in ("ENGAGE, ALERT") #Rimesso ALERT -> segnalazione per revisione umana, non engagement diretto
                if is_threat and is_engaged:
                    self.metrics.tp += 1
                elif not is_threat and is_engaged:
                    self.metrics.fp += 1
                elif is_threat and not is_engaged:
                    self.metrics.fn += 1
                else:
                    self.metrics.tn += 1
                self.metrics.threat_scores.append(fusion_r.threat_score)

                if verbose and entity.role == AgentRole.TARGET:
                    self.step_log.append({
                        "step": self.env.step_idx,
                        "id": entity.id,
                        "action": decision.action,
                        "score": round(fusion_r.threat_score, 4),
                        "patch": vision_d.patch_active,
                        "real_yolo": vision_d.real_yolo,
                        "poisoned": osint_p.is_poisoned
                    })

        return self.metrics
    
def evaluate_on_dataset(loader, patch_tensor=None,
                        model_path: str = "yolov8n.pt",
                        conf_threshold: float = 0.50,
                        max_samples: int = None,
                        verbose: bool = True) -> SimMetrics:
    """
    F1 reale su frame VisDrone — sostituisce il decadimento matematico (fake)
    Logica frame-level: TP se YOLO rileva almeno 1 persona dove ce n'è 1 annotata.
    """
    try:
        import torch
        from ultralytics import YOLO
        from patch_optimizer import PatchOptimizer, get_chest_bbox
        from config import IMG_SIZE, PERSON_CLASS_ID
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(f"Dipendenza mancante: {e}")

    model   = YOLO(model_path)
    metrics = SimMetrics()
    indices = list(range(len(loader)))
    if max_samples:
        indices = indices[:max_samples]

    for n_done, idx in enumerate(indices):
        img_pil, gt_bboxes = loader.get_sample(idx)
        has_person_gt      = len(gt_bboxes) > 0

        # applica patch su ogni persona annotata nel frame
        if patch_tensor is not None and gt_bboxes:
            img_t = torch.from_numpy(
                np.array(img_pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)
            for bbox in gt_bboxes:
                patch_bbox = get_chest_bbox(
                    bbox, patch_tensor.shape[2], patch_tensor.shape[1], IMG_SIZE)
                img_t = PatchOptimizer.apply_patch_to_image(img_t, patch_tensor, patch_bbox)
            img_pil = Image.fromarray(
                (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8))

        # inferenza YOLO
        results      = model(img_pil, verbose=False)
        detected_any = any(
            (r.boxes is not None and
             (r.boxes.cls == PERSON_CLASS_ID).any() and
             r.boxes.conf[(r.boxes.cls == PERSON_CLASS_ID)].max() >= conf_threshold)
            for r in results if r.boxes is not None and len(r.boxes) > 0
        )

        if   has_person_gt and     detected_any: metrics.tp += 1
        elif has_person_gt and not detected_any: metrics.fn += 1
        elif not has_person_gt and detected_any: metrics.fp += 1
        else:                                    metrics.tn += 1

        if verbose and (n_done + 1) % 20 == 0:
            print(f"  {n_done+1}/{len(indices)}  "
                  f"P={metrics.precision:.3f}  R={metrics.recall:.3f}  F1={metrics.f1:.3f}")

    if verbose:
        print(f"\n  Finale: TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} TN={metrics.tn}")
        print(f"  P={metrics.precision:.3f}  R={metrics.recall:.3f}  "
              f"F1={metrics.f1:.3f}  FPR={metrics.fpr:.3f}")
    return metrics