"""
Fusione bayesiana di segnali di visione, OSINT e comportamentali per valutare minacce e prendere decisioni operative con conformità al diritto internazionale umanitario (IHL).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

from config import FUSION_WEIGHTS, ENGAGEMENT_THRESHOLD, ALERT_THRESHOLD, TRACK_THRESHOLD
from detection import VisionDetection
from entities import OSINTProfile


@dataclass
class FusionResult:
    threat_score: float
    vision_contrib: float
    osint_contrib: float
    behavioral_contrib: float
    confidence_interval: Tuple[float, float]


@dataclass
class Decision:
    action: str
    threat_score: float
    ihl_compliant: bool
    rationale: str


class FusionAgent:
    """Faccio la fusione bayesiana tra segnali vision (YOLO), OSINT e comportamentali (behavioral)"""

    PRIOR = 0.50

    def __init__(self):
        self.history: Dict[int, List[float]] = {}
        self.w = FUSION_WEIGHTS

    def _bay(self, prior: float, likelihood: float) -> float:
        p = likelihood * prior
        q = (1 - likelihood) * (1 - prior)
        return float(np.clip(p / (p + q + 1e-9), 0, 1))

    def _ci(self, scores: List[float]) -> Tuple[float, float]:
        if len(scores) < 2:
            return (0.0, 1.0)
        arr = np.array(scores)
        return (
            max(0.0, float(np.mean(arr) - 1.96 * np.std(arr))),
            min(1.0, float(np.mean(arr) + 1.96 * np.std(arr)))
        )

    def fuse(self, eid: int, vision: VisionDetection, osint: OSINTProfile,
             behavioral: float) -> FusionResult:
        v = vision.confidence * (1.0 if vision.detected else 0.5)
        raw = (
            self.w["vision"] * v +
            self.w["osint"] * osint.threat_score +
            self.w["behavioral"] * behavioral
        )
        threat = self._bay(self.PRIOR, raw)

        self.history.setdefault(eid, [])
        self.history[eid].append(threat)
        if len(self.history[eid]) > 20:
            self.history[eid].pop(0)

        return FusionResult(
            threat_score=threat,
            vision_contrib=self.w["vision"] * v,
            osint_contrib=self.w["osint"] * osint.threat_score,
            behavioral_contrib=self.w["behavioral"] * behavioral,
            confidence_interval=self._ci(self.history[eid][-10:])
        )


class DecisionAgent:
    """Logica di decision in base al risultato della fusion e tenendo conto di limitazioni IHL(International Humanitarian Law)"""

    def __init__(self):
        self.log: List[Decision] = []
        self.engagements = 0
        self.ihl_overrides = 0

    def _ihl_ok(self, fusion: FusionResult, nearby_civilians: int) -> bool:
        #troppa variazione nelle fusion recenti -> false positive evito engagement
        if (fusion.confidence_interval[1] - fusion.confidence_interval[0]) > 0.40:
            return False
        #troppi? civili vicini a una minaccia potenziale -> evito engagement
        if nearby_civilians > 3 and fusion.threat_score < 0.85:
            return False
        return True

    def decide(self, eid: int, fusion: FusionResult, nearby_civilians: int = 0) -> Decision:
        ihl = self._ihl_ok(fusion, nearby_civilians)
        score = fusion.threat_score

        # 1) azione nominale, determinata dalle sole soglie
        if score >= ENGAGEMENT_THRESHOLD:
            action = "ENGAGE"
        elif score >= ALERT_THRESHOLD:
            action = "ALERT"
        elif score >= TRACK_THRESHOLD:
            action = "TRACK"
        else:
            action = "IGNORE"

        # 2) veto IHL: declassamento a supervisione umana
        if action == "ENGAGE" and not ihl:
            action = "ALERT"
            self.ihl_overrides += 1

        if action == "ENGAGE":
            self.engagements += 1

        rationale = (
            f"score={score:.3f} CI=[{fusion.confidence_interval[0]:.2f},"
            f"{fusion.confidence_interval[1]:.2f}] "
            f"V={fusion.vision_contrib:.3f} O={fusion.osint_contrib:.3f} "
            f"B={fusion.behavioral_contrib:.3f}"
        )
        decision = Decision(
            action=action,
            threat_score=score,
            ihl_compliant=ihl,
            rationale=rationale
        )
        self.log.append(decision)
        return decision