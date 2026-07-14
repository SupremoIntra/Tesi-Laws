"""
LAWS-SIM loop del simulator.
"""

import random
import numpy as np
import json
import os
import cv2
from typing import Tuple, List, Dict
from PIL import Image

from config import GRID_SIZE, VISION_METRICS_JSON
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
        if os.path.exists(VISION_METRICS_JSON):
            with open(VISION_METRICS_JSON, "r") as f:
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
                is_engaged = decision.action in ("ENGAGE", "ALERT") #Rimesso ALERT -> segnalazione per revisione umana, non engagement diretto
                
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
                        "poisoned": osint_p.is_poisoned
                    })

        return self.metrics
    
def evaluate_on_dataset(loader, patch_tensor=None,
                        model_path: str = "yolov8n.pt",
                        conf_threshold: float = 0.50,
                        max_samples: int = None,
                        verbose: bool = True,
                        tactical_min_height: float = 80.0) -> Tuple[SimMetrics, SimMetrics, float, List[Dict[str, int]]]:
    """
    F1 reale su frame VisDrone — sostituisce il decadimento matematico (fake)
    Logica frame-level: TP se YOLO rileva almeno 1 persona dove ce n'è 1 annotata.
    Con iniezione corretta della patch proporzionale.

    # TACTICAL FILTER 2026
    Calcola, nello STESSO loop di inferenza (zero costo aggiuntivo), anche
    una seconda metrica ristretta ai soli target "tatticamente rilevanti"
    (altezza >= tactical_min_height, default 80px, coerente con lo scenario
    del simulatore a DRONE_ALTITUDE_M=10). Ritorna anche tactical_coverage:
    percentuale di annotazioni valide (>=60px) che sono anche >=80px.

    # BOOTSTRAP CI (richiesta relatore)
    Ritorna anche `per_frame_outcomes`: lista di dict {"tp","fp","tn","fn"}
    (uno-hot, un solo campo a 1 per frame), stessa unità che viene poi
    ricampionata da `metrics.bootstrap_ci`. Riferita alla metrica
    COMPLETA (non quella tattica) — è quella su cui il relatore ha
    chiesto gli intervalli di confidenza.

    Returns:
        (metrics_completo, metrics_filtrato_tattico, tactical_coverage, per_frame_outcomes)
    """
    try:
        from ultralytics import YOLO
        from patch_optimizer import get_chest_bbox_proportional
        from config import PERSON_CLASS_ID
    except ImportError as e:
        raise RuntimeError(f"Dipendenza mancante: {e}")

    model   = YOLO(model_path)
    metrics = SimMetrics()
    metrics_tactical = SimMetrics()  # TACTICAL FILTER 2026
    n_annotations_valid = 0          # TACTICAL FILTER 2026 (>=60px, come sempre)
    n_annotations_tactical = 0       # TACTICAL FILTER 2026 (>=tactical_min_height)
    per_frame_outcomes: List[Dict[str, int]] = []  # BOOTSTRAP CI
    indices = list(range(len(loader)))
    if max_samples:
        indices = indices[:max_samples]

    patch_img_cv = None
    if patch_tensor is not None:
        patch_img_cv = patch_tensor.squeeze(0).permute(1, 2, 0).numpy()
        patch_img_cv = (patch_img_cv * 255).astype(np.uint8)

    for n_done, idx in enumerate(indices):
        img_pil, gt_bboxes = loader.get_sample(idx)
        
        # Filtriamo le ground truth valide (> 60 pixel di altezza) per evitare downsampling distruttivo
        valid_gt_bboxes = [b for b in gt_bboxes if (b[3] - b[1]) >= 60]
        has_person_gt = len(valid_gt_bboxes) > 0

        # BOOTSTRAP CI / specificity: un frame senza persone valide (<60px)
        # non e' automaticamente "negativo" in senso stretto -- potrebbe
        # avere persone reali solo troppo piccole per essere un target
        # valido. Serve la classe negativa VERA (zero persone di
        # qualunque dimensione) per calcolare specificity in modo
        # corretto; un frame "ambiguo" (solo persone sotto soglia) viene
        # escluso da entrambe le classi, non forzato in una delle due.
        is_truly_empty = len(gt_bboxes) == 0

        if not has_person_gt and not is_truly_empty:
            continue  # ambiguo: solo persone <60px, ne' positivo ne' negativo pulito

        # TACTICAL FILTER 2026: sottoinsieme "tatticamente rilevante" del frame
        tactical_gt_bboxes = [b for b in valid_gt_bboxes if (b[3] - b[1]) >= tactical_min_height]
        has_person_gt_tactical = len(tactical_gt_bboxes) > 0
        n_annotations_valid += len(valid_gt_bboxes)
        n_annotations_tactical += len(tactical_gt_bboxes)

        if patch_img_cv is not None:
            img_cv = np.array(img_pil)
            img_h, img_w = img_cv.shape[:2]
            
            for bbox in valid_gt_bboxes:
                px1, py1, px2, py2 = get_chest_bbox_proportional(bbox, img_w, img_h)
                eff_pw, eff_ph = px2 - px1, py2 - py1
                if eff_pw > 0 and eff_ph > 0:
                    patch_resized = cv2.resize(patch_img_cv, (eff_pw, eff_ph), interpolation=cv2.INTER_AREA)
                    img_cv[py1:py2, px1:px2] = patch_resized
            img_pil = Image.fromarray(img_cv)

        # inferenza YOLO
        results = model(img_pil, verbose=False)
        detected_any = any(
            (r.boxes is not None and
             (r.boxes.cls == PERSON_CLASS_ID).any() and
             r.boxes.conf[(r.boxes.cls == PERSON_CLASS_ID)].max() >= conf_threshold)
            for r in results if r.boxes is not None and len(r.boxes) > 0
        )

        if has_person_gt and detected_any: metrics.tp += 1
        elif has_person_gt and not detected_any: metrics.fn += 1
        elif not has_person_gt and detected_any: metrics.fp += 1
        else: metrics.tn += 1

        # BOOTSTRAP CI: esito di questo frame, uno-hot, per il ricampionamento
        per_frame_outcomes.append({
            "tp": int(has_person_gt and detected_any),
            "fn": int(has_person_gt and not detected_any),
            "fp": int((not has_person_gt) and detected_any),
            "tn": int((not has_person_gt) and (not detected_any)),
        })

        # TACTICAL FILTER 2026: stessa detection (detected_any), ma frame
        # conteggiato solo se ha almeno un target >= tactical_min_height
        if has_person_gt_tactical and detected_any: metrics_tactical.tp += 1
        elif has_person_gt_tactical and not detected_any: metrics_tactical.fn += 1
        elif not has_person_gt_tactical and detected_any: metrics_tactical.fp += 1
        else: metrics_tactical.tn += 1

        if verbose and (n_done + 1) % 20 == 0:
            print(f"  {n_done+1}/{len(indices)}  "
                  f"P={metrics.precision:.3f}  R={metrics.recall:.3f}  F1={metrics.f1:.3f}")

    tactical_coverage = (n_annotations_tactical / n_annotations_valid) if n_annotations_valid > 0 else 0.0

    if verbose:
        print(f"\n  Finale: TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} TN={metrics.tn}")
        print(f"  P={metrics.precision:.3f}  R={metrics.recall:.3f}  "
              f"F1={metrics.f1:.3f}  FPR={metrics.fpr:.3f}")
        print(f"\n  [TACTICAL FILTER >= {tactical_min_height:.0f}px] "
              f"TP={metrics_tactical.tp} FP={metrics_tactical.fp} "
              f"FN={metrics_tactical.fn} TN={metrics_tactical.tn}")
        print(f"  Evasion Rate tattico: "
              f"{(metrics_tactical.fn / max(metrics_tactical.tp + metrics_tactical.fn, 1)) * 100:.1f}%  "
              f"| Copertura tattica del valset: {tactical_coverage*100:.1f}%")

    return metrics, metrics_tactical, tactical_coverage, per_frame_outcomes