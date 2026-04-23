"""
LAWS-SIM core simulation loop.
"""

import random
import numpy as np

from config import GRID_SIZE
from entities import Environment, AgentRole
from detection import VisionAgentReal
from fusion_decision import FusionAgent, DecisionAgent
from metrics import SimMetrics, AttackScenario


class LAWSSim:
    """Multi-agent LAWS simulator."""

    def __init__(self, scenario: AttackScenario, steps: int, seed: int = 42,
                 real_mode: bool = False, image_dir: str = None,
                 patch_tensor: "torch.Tensor" = None):
        random.seed(seed)
        np.random.seed(seed)

        self.scenario = scenario
        self.steps = steps
        self.env = Environment(grid_size=GRID_SIZE)
        self.osint_agent = None  # (simplified, embedded in SimEntity)
        self.vision = VisionAgentReal(
            real_mode=real_mode,
            image_dir=image_dir,
            patch_tensor=patch_tensor
        )
        self.fusion = FusionAgent()
        self.decision = DecisionAgent()
        self.metrics = SimMetrics()
        self.step_log = []

        # Activate CARE Kit for targets in patch scenarios
        if scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING):
            for e in self.env.entities:
                if e.role == AgentRole.TARGET:
                    e.care_kit_active = True

    def run(self, verbose: bool = False) -> SimMetrics:
        patch_on = self.scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING)
        poison_on = self.scenario in (AttackScenario.OSINT_POISON, AttackScenario.CASCADING)

        for _ in range(self.steps):
            self.env.tick()
            for entity in self.env.visible():
                dist = self.env.dist_to_drone(entity)
                civ_near = self.env.nearby_civilians(entity)

                # OSINT analysis (attacco di poisoning... se on)
                osint_p = entity.osint_profile
                if poison_on and entity.role == AgentRole.TARGET:
                    #abbasso punteggi di minaccia del target per confondere la fusione
                    #random.uniform -> numero casuale tra a e b (valori scelti per simulare... in entities ho i limiti di normali, es non negativo)
                    osint_p.social_score = max(0.0, osint_p.social_score - random.uniform(0.40, 0.60))
                    osint_p.geo_anomaly = max(0.0, osint_p.geo_anomaly - random.uniform(0.30, 0.55))
                    osint_p.network_centrality = max(0.0, osint_p.network_centrality - random.uniform(0.25, 0.45))
                    osint_p.is_poisoned = True

                vision_d = self.vision.detect(entity, dist, patch_active=patch_on)
                fusion_r = self.fusion.fuse(
                    entity.id, vision_d, osint_p, entity.behavioral_score
                )
                decision = self.decision.decide(entity.id, fusion_r, civ_near)

                # Aggiorno le metriche 
                is_threat = (entity.role == AgentRole.TARGET)
                is_engaged = decision.action in ("ENGAGE", "ALERT")
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