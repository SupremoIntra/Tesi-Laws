"""
Simulation metrics and CLAE calculation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from config import PATCH_BBOX_COVERAGE, OSINT_FIELDS_TOTAL, OSINT_FIELDS_POISONED


class AttackScenario(Enum):
    NONE = "Baseline (No Attack)"
    PATCH_ONLY = "Adversarial Patch [Vision]"
    OSINT_POISON = "OSINT Poisoning"
    CASCADING = "Cascading Attack [Multi-Layer]"


@dataclass
class SimMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    threat_scores: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0


def clae_costs() -> Dict[str, Optional[float]]:
    """
    Compute CLAE costs from physically measurable quantities.

    C_vision = patch_area / bbox_area (pixel ratio)
    C_osint  = fields_poisoned / fields_total
    C_cascading = 1 - (1 - C_v)(1 - C_o)  (union probability)
    """
    c_v = PATCH_BBOX_COVERAGE
    c_o = OSINT_FIELDS_POISONED / OSINT_FIELDS_TOTAL
    c_c = 1.0 - (1.0 - c_v) * (1.0 - c_o)
    return {
        "NONE": None,
        "PATCH_ONLY": c_v,
        "OSINT_POISON": c_o,
        "CASCADING": c_c
    }


def compute_clae(scenario: AttackScenario, attack_m: SimMetrics,
                 baseline_m: SimMetrics) -> Optional[float]:
    """CLAE = ΔF1 / C_attack"""
    costs = clae_costs()
    cost = costs.get(scenario.name)
    if cost is None:
        return None
    return (baseline_m.f1 - attack_m.f1) / cost if cost > 0 else 0.0