"""
Simulation metrics and CLAE calculation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from config import PATCH_BBOX_COVERAGE, OSINT_FIELDS_TOTAL, OSINT_FIELDS_POISONED


class AttackScenario(Enum):
    BASELINE = "Baseline (No Attack)"
    PATCH_ONLY = "Adversarial Patch [Vision]"
    OSINT_POISONING = "OSINT Poisoning"
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

    # === Metriche proposte dal relatore (call di luglio) ===

    @property
    def sensitivity(self) -> float:
        """
        R1 = recall sui positivi = TP/(TP+FN).
        Tra le immagini CON pedone, quante ne rileva YOLO. Sotto attacco
        efficace deve SCENDERE (il pedone sparisce alla detection).
        Sinonimo di `recall`, esplicitato con il nome clinico standard
        per allinearsi alla terminologia richiesta dal relatore.
        """
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        """
        R2 = recall sui negativi = TN/(TN+FP).
        Tra le immagini SENZA pedone, quante YOLO classifica
        correttamente come vuote. Non deve crollare: se l'attacco
        inducesse falsi positivi altrove, R2 scenderebbe e la media
        geometrica lo penalizzerebbe.
        """
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def geometric_mean_recall(self) -> float:
        """
        Metrica primaria proposta dal relatore: sqrt(R1 * R2), media
        geometrica di sensitivity e specificity.

        Il prodotto di due valori in [0,1] è sempre <= del più piccolo
        dei due: per far salire la media geometrica DEVONO salire
        entrambi i recall, e devono essere vicini tra loro. Impedisce di
        "barare" gonfiando un recall a scapito dell'altro — premia un
        comportamento del detector forte E bilanciato.
        """
        return geometric_mean_recall(self.sensitivity, self.specificity)


def geometric_mean_recall(r1: float, r2: float) -> float:
    """
    Media geometrica dei due recall: sqrt(R1 * R2).

    Args:
        r1: recall sui positivi (sensitivity), in [0,1]
        r2: recall sui negativi (specificity), in [0,1]

    Returns:
        sqrt(r1 * r2), in [0,1]. Vale 0 se uno dei due è 0.
    """
    r1 = max(0.0, min(1.0, r1))
    r2 = max(0.0, min(1.0, r2))
    return float((r1 * r2) ** 0.5)


def sensitivity_specificity(tp: int, fn: int, tn: int, fp: int) -> Tuple[float, float]:
    """
    Calcola (sensitivity, specificity) da conteggi grezzi.

    sensitivity = TP/(TP+FN) = R1 (recall sui positivi)
    specificity = TN/(TN+FP) = R2 (recall sui negativi)

    Utile fuori dalla dataclass SimMetrics, es. nel bootstrap dove si
    ricampionano direttamente i conteggi.
    """
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sens), float(spec)


def bootstrap_ci(
    per_sample_outcomes: List[Dict[str, int]],
    metric_fn,
    n_iter: int = 10000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Intervallo di confidenza via bootstrap con metodo dei percentili.

    Metodo (come indicato dal relatore): si ricampiona il set di test
    n_iter volte CON reinserimento, ogni volta si ricalcola la metrica,
    e l'intervallo al 95% sono il 2.5° e 97.5° percentile della
    distribuzione dei valori ottenuti. Serve a stabilire se una
    differenza (es. F1 pre vs post attacco) è statisticamente
    significativa: se gli intervalli pre e post si sovrappongono
    ampiamente, la differenza potrebbe non essere reale.

    Args:
        per_sample_outcomes: lista di dict, uno per frame di test, con
            chiavi "tp","fp","tn","fn" (ognuna 0 o 1 a livello frame).
            È l'unità che viene ricampionata.
        metric_fn: funzione che prende (tp,fp,tn,fn) aggregati e ritorna
            un float (es. lambda t,f,n,fn: geometric_mean_recall(...)).
        n_iter: numero di ricampionamenti bootstrap (default 10000).
        ci: livello di confidenza (default 0.95 → percentili 2.5/97.5).
        seed: per riproducibilità.

    Returns:
        dict con "point" (valore sul campione originale), "low", "high"
        (estremi dell'intervallo), "mean" (media bootstrap).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(per_sample_outcomes)
    if n == 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "mean": 0.0}

    tp = np.array([o.get("tp", 0) for o in per_sample_outcomes])
    fp = np.array([o.get("fp", 0) for o in per_sample_outcomes])
    tn = np.array([o.get("tn", 0) for o in per_sample_outcomes])
    fn = np.array([o.get("fn", 0) for o in per_sample_outcomes])

    # Valore puntuale sul campione originale
    point = metric_fn(int(tp.sum()), int(fp.sum()), int(tn.sum()), int(fn.sum()))

    boot_vals = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)  # ricampionamento con reinserimento
        boot_vals[i] = metric_fn(
            int(tp[idx].sum()), int(fp[idx].sum()),
            int(tn[idx].sum()), int(fn[idx].sum())
        )

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_vals, alpha * 100))
    high = float(np.percentile(boot_vals, (1.0 - alpha) * 100))
    return {
        "point": float(point),
        "low": low,
        "high": high,
        "mean": float(boot_vals.mean()),
    }


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


def compute_clae(metrics, c_vision: float, c_osint: float) -> float:
    """
    Metrica CEAE (Cost-Effective Adversarial Engagement).
    Formula accademica: penalizza FP (danni collaterali) e FN (evasione) 
    aggravandoli con il costo specifico dell'attacco in corso.
    """
    # Se non c'è nulla da valutare, la metrica è 0
    if (metrics.tp + metrics.fp + metrics.fn) == 0:
        return 0.0

    # Calcolo dei pesi di penalità (Attacco OSINT aggrava i civili colpiti, Vision aggrava le evasioni)
    penalty_fp = metrics.fp * (1.0 + c_osint)
    penalty_fn = metrics.fn * (1.0 + c_vision)
    
    denominator = metrics.tp + penalty_fp + penalty_fn
    
    if denominator == 0:
        return 0.0
        
    return float(metrics.tp / denominator)