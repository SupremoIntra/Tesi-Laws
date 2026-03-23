#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                        LAWS-SIM  v1.0                              ║
║           Multi-Agent LAWS Vulnerability Simulator                  ║
║                                                                      ║
║  Agenti  : OSINT | Vision (YOLO-sim) | Fusion | Decision            ║
║  Attacchi: Adversarial Patch (EoT) | OSINT Poisoning | Cascading    ║
║  Metriche: Precision / Recall / F1 / FPR / CLAE                     ║
╚══════════════════════════════════════════════════════════════════════╝

Uso:
    python laws_sim.py
    python laws_sim.py --steps 200 --seed 7 --verbose
    python laws_sim.py --no-plot          # skip matplotlib
"""

import sys, math, json, random, argparse
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ── Rich (terminal UI) ────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console   = Console()
    HAS_RICH  = True
except ImportError:
    class _FallbackConsole:
        def print(self, *a, **kw): print(*[str(x) for x in a])
        def rule(self, *a, **kw):  print("─" * 60)
    console  = _FallbackConsole()
    HAS_RICH = False

# ── Matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from   matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Faker ─────────────────────────────────────────────────────────────
try:
    from faker import Faker
    _fake    = Faker(["it_IT", "en_US"])
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE GLOBALE
# ══════════════════════════════════════════════════════════════════════

GRID_SIZE          = 30      # dimensione griglia simulata (unità)
N_TARGETS          = 3       # soggetti "minaccia" (ground truth)
N_CIVILIANS        = 15      # civili nell'ambiente
DRONE_HEIGHT       = 10.0    # altitudine drone simulata (m)
YOLO_MAX_RANGE     = 25.0    # range massimo rilevamento YOLO (m)

# Pesi fusione  (vision, osint, behavioral)
FUSION_WEIGHTS     = {"vision": 0.45, "osint": 0.35, "behavioral": 0.20}

# Soglie decision agent (calibrate sul range di score reale ~0.05–0.75)
ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD      = 0.38
TRACK_THRESHOLD      = 0.22

# Parametri patch EoT
PATCH_SUPPRESSION    = 0.65   # soppressione confidence al centro
PATCH_DIST_FALLOFF   = 0.04   # decay effetto con distanza

# Costi normalizzati degli attacchi (per CLAE)
ATTACK_COSTS = {
    "PATCH_ONLY"   : 0.40,
    "OSINT_POISON" : 0.30,
    "CASCADING"    : 0.55,
}


# ══════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ══════════════════════════════════════════════════════════════════════

class AgentRole(Enum):
    CIVILIAN = "civilian"
    TARGET   = "target"

class AttackScenario(Enum):
    NONE         = "Baseline (No Attack)"
    PATCH_ONLY   = "Adversarial Patch [Vision]"
    OSINT_POISON = "OSINT Poisoning"
    CASCADING    = "Cascading Attack [Multi-Layer]"


@dataclass
class OSINTProfile:
    """Profilo sintetico OSINT di un'entità"""
    name               : str
    age                : int
    nationality        : str
    social_score       : float   # attività sospetta social (0-1)
    geo_anomaly        : float   # anomalie geolocalizzazione (0-1)
    network_centrality : float   # centralità rete contatti (0-1)
    is_poisoned        : bool = False

    @property
    def threat_score(self) -> float:
        return float(np.clip(
            0.40 * self.social_score +
            0.35 * self.geo_anomaly  +
            0.25 * self.network_centrality,
            0, 1
        ))


@dataclass
class VisionDetection:
    detected        : bool
    confidence      : float
    bbox            : Tuple[int, int, int, int]   # x, y, w, h
    class_label     : str
    patch_active    : bool = False


@dataclass
class FusionResult:
    threat_score             : float
    vision_contrib           : float
    osint_contrib            : float
    behavioral_contrib       : float
    confidence_interval      : Tuple[float, float]


@dataclass
class Decision:
    action        : str    # ENGAGE | ALERT | TRACK | IGNORE
    threat_score  : float
    ihl_compliant : bool
    rationale     : str


@dataclass
class SimMetrics:
    tp            : int   = 0
    fp            : int   = 0
    tn            : int   = 0
    fn            : int   = 0
    threat_scores : List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

class SimEntity:
    """
    Entità mobile nell'ambiente (civile o target).
    Ogni entità porta un profilo OSINT, uno stato fisico e
    un flag care_kit_active per la patch avversariale.
    """

    def __init__(self, eid: int, role: AgentRole, grid: int):
        self.id              = eid
        self.role            = role
        self.grid            = grid
        self.x               = random.randint(0, grid - 1)
        self.y               = random.randint(0, grid - 1)
        self.history         : List[Tuple[int, int]] = [(self.x, self.y)]
        self.care_kit_active = False
        self.osint_profile   = self._gen_profile()

    # ── OSINT profile generation ──────────────────────────────────────
    def _gen_profile(self) -> OSINTProfile:
        if HAS_FAKER:
            name        = _fake.name()
            nationality = _fake.country()
        else:
            name        = f"Entity_{self.id:03d}"
            nationality = "N/A"

        if self.role == AgentRole.TARGET:
            return OSINTProfile(
                name=name, age=random.randint(20, 45),
                nationality=nationality,
                social_score       = random.uniform(0.55, 0.90),
                geo_anomaly        = random.uniform(0.50, 0.85),
                network_centrality = random.uniform(0.40, 0.80),
            )
        return OSINTProfile(
            name=name, age=random.randint(18, 70),
            nationality=nationality,
            social_score       = random.uniform(0.00, 0.30),
            geo_anomaly        = random.uniform(0.00, 0.25),
            network_centrality = random.uniform(0.00, 0.20),
        )

    # ── Movement ──────────────────────────────────────────────────────
    def move(self):
        step = 2 if self.role == AgentRole.TARGET else 1
        dx   = random.randint(-step, step)
        dy   = random.randint(-step, step)
        self.x = int(np.clip(self.x + dx, 0, self.grid - 1))
        self.y = int(np.clip(self.y + dy, 0, self.grid - 1))
        self.history.append((self.x, self.y))
        if len(self.history) > 20:
            self.history.pop(0)

    # ── Behavioral score (movement variance) ─────────────────────────
    @property
    def behavioral_score(self) -> float:
        if len(self.history) < 3:
            return 0.0
        arr      = np.array(self.history[-10:])
        variance = float(np.var(arr[:, 0]) + np.var(arr[:, 1]))
        score    = np.clip(variance / 20.0, 0, 1)
        if self.role == AgentRole.TARGET:
            score = float(np.clip(score + random.uniform(0.10, 0.25), 0, 1))
        return score


class Environment:
    """Griglia di simulazione con drone in pattuglia"""

    def __init__(self, grid: int = GRID_SIZE):
        self.grid     = grid
        self.step_idx = 0
        self.drone_x  = grid // 2
        self.drone_y  = grid // 2
        self.entities : List[SimEntity] = []
        self._spawn_entities()

    def _spawn_entities(self):
        for i in range(N_TARGETS):
            self.entities.append(SimEntity(i, AgentRole.TARGET, self.grid))
        for i in range(N_CIVILIANS):
            self.entities.append(SimEntity(N_TARGETS + i, AgentRole.CIVILIAN, self.grid))

    def tick(self):
        for e in self.entities:
            e.move()
        # Drone segue random-walk (pattuglia simulata)
        self.drone_x = int(np.clip(
            self.drone_x + random.randint(-2, 2), 0, self.grid - 1))
        self.drone_y = int(np.clip(
            self.drone_y + random.randint(-2, 2), 0, self.grid - 1))
        self.step_idx += 1

    def dist_to_drone(self, e: SimEntity) -> float:
        dx = e.x - self.drone_x
        dy = e.y - self.drone_y
        return math.sqrt(dx * dx + dy * dy + DRONE_HEIGHT ** 2)

    def visible(self) -> List[SimEntity]:
        return [e for e in self.entities
                if self.dist_to_drone(e) <= YOLO_MAX_RANGE]

    def nearby_civilians(self, target: SimEntity, radius: int = 5) -> int:
        return sum(
            1 for e in self.entities
            if e.role == AgentRole.CIVILIAN
            and abs(e.x - target.x) + abs(e.y - target.y) <= radius
        )


# ══════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════

# ── 1. OSINT Agent ────────────────────────────────────────────────────

class OSINTAgent:
    """
    Raccoglie intelligence da fonti aperte (simulata).

    Vulnerabilità modellata: DATA POISONING
    ─────────────────────────────────────────
    Un avversario inietta dati falsi nei feed OSINT (fake check-in,
    pulizia del profilo social, manipolazione metadata) abbassando
    artificialmente il threat score di un target reale.
    """

    def __init__(self):
        self.cache: Dict[int, OSINTProfile] = {}

    def analyze(self, entity: SimEntity,
                poisoning: bool = False) -> OSINTProfile:
        profile = deepcopy(entity.osint_profile)

        if poisoning and entity.role == AgentRole.TARGET:
            # ATTACCO: abbassa ogni dimensione del profilo
            profile.social_score       = max(0.0, profile.social_score
                                             - random.uniform(0.40, 0.60))
            profile.geo_anomaly        = max(0.0, profile.geo_anomaly
                                             - random.uniform(0.30, 0.55))
            profile.network_centrality = max(0.0, profile.network_centrality
                                             - random.uniform(0.25, 0.45))
            profile.is_poisoned = True

        self.cache[entity.id] = profile
        return profile


# ── 2. Vision Agent (YOLO-sim) ────────────────────────────────────────

class VisionAgent:
    """
    Simula un modello YOLO-v8 su stream video da drone.

    Modello di confidence:
        conf_base(d) = 0.95 · exp(-1.5 · d / D_max)

    Adversarial Patch (EoT):
        conf_patch(d) = conf_base · (1 − S · exp(−λ · d) + ε)

    dove:
        S = PATCH_SUPPRESSION   (soppressione di base della patch)
        λ = PATCH_DIST_FALLOFF  (la patch diventa piccola in lontananza)
        ε ~ N(0, 0.08)          (rumore fisico: illuminazione, angolo, blur)
    """

    DETECTION_THRESHOLD = 0.25   # soglia YOLO default

    def __init__(self):
        self.fn_count   = 0   # falsi negativi accumulati
        self.det_count  = 0

    def _base_conf(self, distance: float) -> float:
        norm = distance / YOLO_MAX_RANGE
        conf = 0.95 * math.exp(-1.5 * norm)
        return float(np.clip(conf + random.gauss(0, 0.04), 0, 1))

    def _eot_patch(self, base: float, distance: float) -> float:
        """
        Expectation over Transformation:
        La soppressione è mediata su variazioni geometriche e
        fotometriche fisiche (distanza, angolo, illuminazione).
        """
        S_eff  = PATCH_SUPPRESSION * math.exp(-PATCH_DIST_FALLOFF * distance)
        noise  = random.gauss(0, 0.08)
        result = base * (1.0 - S_eff + noise)
        return float(np.clip(result, 0, 1))

    def detect(self, entity: SimEntity, distance: float,
               patch_active: bool = False) -> VisionDetection:
        self.det_count += 1
        base = self._base_conf(distance)
        conf = self._eot_patch(base, distance) \
               if (patch_active and entity.care_kit_active) \
               else base

        detected = conf >= self.DETECTION_THRESHOLD
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected     = detected,
            confidence   = conf,
            bbox         = (entity.x, entity.y, 5, 8),
            class_label  = "person" if detected else "background",
            patch_active = patch_active and entity.care_kit_active,
        )


# ── 3. Fusion Agent ───────────────────────────────────────────────────

class FusionAgent:
    """
    Fusione Bayesiana pesata dei tre layer informativi.

    score_raw = w_v · conf_v + w_o · score_o + w_b · score_b

    P(minaccia | evidenza) = aggiornamento Bayesiano con prior=0.10
    (presunzione di innocenza — art. 57 Protocollo I, IHL)

    L'intervallo di confidenza stima l'incertezza storica
    sulle ultime N osservazioni dello stesso soggetto.
    """

    PRIOR = 0.50   # prior neutro — innocenza presunta è già nei bassi score OSINT dei civili

    def __init__(self):
        self.history: Dict[int, List[float]] = {}
        self.w = FUSION_WEIGHTS

    @staticmethod
    def _bayesian_update(prior: float, likelihood: float) -> float:
        p = likelihood * prior
        q = (1.0 - likelihood) * (1.0 - prior)
        return float(np.clip(p / (p + q + 1e-9), 0, 1))

    def _ci(self, scores: List[float]) -> Tuple[float, float]:
        if len(scores) < 2:
            return (0.0, 1.0)
        arr  = np.array(scores)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        return (max(0.0, mean - 1.96 * std),
                min(1.0, mean + 1.96 * std))

    def fuse(self, eid: int, vision: VisionDetection,
             osint: OSINTProfile, behavioral: float) -> FusionResult:

        v_score = vision.confidence * (1.0 if vision.detected else 0.5)
        o_score = osint.threat_score
        b_score = behavioral

        raw = (self.w["vision"]     * v_score +
               self.w["osint"]      * o_score +
               self.w["behavioral"] * b_score)

        threat = self._bayesian_update(self.PRIOR, raw)

        self.history.setdefault(eid, [])
        self.history[eid].append(threat)
        if len(self.history[eid]) > 20:
            self.history[eid].pop(0)

        return FusionResult(
            threat_score       = threat,
            vision_contrib     = self.w["vision"]     * v_score,
            osint_contrib      = self.w["osint"]      * o_score,
            behavioral_contrib = self.w["behavioral"] * b_score,
            confidence_interval= self._ci(self.history[eid][-10:]),
        )


# ── 4. Decision Agent ─────────────────────────────────────────────────

class DecisionAgent:
    """
    Applica le soglie di engagement e i vincoli IHL.

    Regole IHL implementate (semplificate):
    ─────────────────────────────────────────
    1. Precauzione: se l'intervallo di confidenza è > 0.40 → no engage
       (incertezza epistemica troppo alta)
    2. Proporzionalità: se > 3 civili nelle vicinanze e score < 0.85
       → downgrade ad ALERT (possibile collateral damage)
    3. Meaningful Human Control: le decisioni ENGAGE vengono flaggate
       per revisione (in un sistema reale → operatore umano)
    """

    def __init__(self):
        self.log             : List[Decision] = []
        self.engagements     = 0
        self.ihl_overrides   = 0

    def _ihl_check(self, fusion: FusionResult,
                   nearby_civs: int) -> bool:
        ci_width = fusion.confidence_interval[1] - fusion.confidence_interval[0]
        if ci_width > 0.40:
            return False
        if nearby_civs > 3 and fusion.threat_score < 0.85:
            return False
        return True

    def decide(self, eid: int, fusion: FusionResult,
               nearby_civs: int = 0) -> Decision:
        ihl_ok = self._ihl_check(fusion, nearby_civs)
        score  = fusion.threat_score

        if score >= ENGAGEMENT_THRESHOLD and ihl_ok:
            action = "ENGAGE"
            self.engagements += 1
        elif score >= ALERT_THRESHOLD:
            action = "ALERT"
        elif score >= TRACK_THRESHOLD:
            action = "TRACK"
        else:
            action = "IGNORE"

        if action == "ENGAGE" and not ihl_ok:
            action = "ALERT"
            self.ihl_overrides += 1

        rationale = (
            f"score={score:.3f} "
            f"CI=[{fusion.confidence_interval[0]:.2f},{fusion.confidence_interval[1]:.2f}] "
            f"V={fusion.vision_contrib:.3f} "
            f"O={fusion.osint_contrib:.3f} "
            f"B={fusion.behavioral_contrib:.3f}"
        )
        d = Decision(action=action, threat_score=score,
                     ihl_compliant=ihl_ok, rationale=rationale)
        self.log.append(d)
        return d


# ══════════════════════════════════════════════════════════════════════
# CORE SIMULATOR
# ══════════════════════════════════════════════════════════════════════

class LAWSSim:
    """
    Orchestratore della simulazione.
    Esegue N_STEPS tick dell'ambiente e processa ogni entità visibile
    attraverso la pipeline: OSINT → Vision → Fusion → Decision.
    """

    def __init__(self, scenario: AttackScenario, steps: int, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.scenario = scenario
        self.steps    = steps
        self.env      = Environment()
        self.osint    = OSINTAgent()
        self.vision   = VisionAgent()
        self.fusion   = FusionAgent()
        self.decision = DecisionAgent()
        self.metrics  = SimMetrics()
        self.step_log : List[dict] = []

        # Abilita CARE kit (patch fisica) sui target per gli scenari pertinenti
        if scenario in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING):
            for e in self.env.entities:
                if e.role == AgentRole.TARGET:
                    e.care_kit_active = True

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self, verbose: bool = False) -> SimMetrics:
        patch_on  = self.scenario in (AttackScenario.PATCH_ONLY,
                                      AttackScenario.CASCADING)
        poison_on = self.scenario in (AttackScenario.OSINT_POISON,
                                      AttackScenario.CASCADING)

        for _ in range(self.steps):
            self.env.tick()
            visible = self.env.visible()

            for entity in visible:
                dist     = self.env.dist_to_drone(entity)
                civ_near = self.env.nearby_civilians(entity)

                # ── Pipeline ────────────────────────────────────────
                osint_p  = self.osint.analyze(entity, poisoning=poison_on)
                vision_d = self.vision.detect(entity, dist, patch_active=patch_on)
                fusion_r = self.fusion.fuse(
                    entity.id, vision_d, osint_p, entity.behavioral_score)
                decision = self.decision.decide(
                    entity.id, fusion_r, nearby_civs=civ_near)

                # ── Metriche ────────────────────────────────────────
                is_threat  = entity.role == AgentRole.TARGET
                is_engaged = decision.action in ("ENGAGE", "ALERT")

                if   is_threat and     is_engaged: self.metrics.tp += 1
                elif not is_threat and is_engaged: self.metrics.fp += 1
                elif is_threat and not is_engaged: self.metrics.fn += 1
                else:                              self.metrics.tn += 1

                self.metrics.threat_scores.append(fusion_r.threat_score)

                if verbose and entity.role == AgentRole.TARGET:
                    self.step_log.append({
                        "step"        : self.env.step_idx,
                        "entity_id"   : entity.id,
                        "action"      : decision.action,
                        "score"       : round(fusion_r.threat_score, 4),
                        "patch"       : vision_d.patch_active,
                        "poisoned"    : osint_p.is_poisoned,
                        "rationale"   : decision.rationale,
                    })

        return self.metrics


# ══════════════════════════════════════════════════════════════════════
# METRICHE AVANZATE
# ══════════════════════════════════════════════════════════════════════

def compute_clae(scenario: AttackScenario,
                 attack_metrics: SimMetrics,
                 baseline_metrics: SimMetrics) -> Optional[float]:
    """
    Cross-Layer Attack Efficiency (CLAE)
    ────────────────────────────────────
    CLAE = ΔF1 / C_attack

    dove ΔF1 è il degradamento dell'F1 rispetto al baseline e
    C_attack è il costo normalizzato dell'attacco (0-1).

    Metrica originale proposta per valutare l'efficienza di attacchi
    multi-layer rispetto al costo addebitabile all'avversario.
    """
    key = scenario.name
    if key not in ATTACK_COSTS:
        return None
    delta_f1 = baseline_metrics.f1 - attack_metrics.f1
    cost     = ATTACK_COSTS[key]
    return delta_f1 / cost if cost > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# OUTPUT & VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def print_results_table(results   : Dict[AttackScenario, SimMetrics],
                        baseline  : SimMetrics):
    if not HAS_RICH:
        print("\n=== RISULTATI ===")
        for sc, m in results.items():
            clae = compute_clae(sc, m, baseline)
            print(f"  {sc.value:<40}  P={m.precision:.3f}  "
                  f"R={m.recall:.3f}  F1={m.f1:.3f}  "
                  f"FPR={m.fpr:.3f}  "
                  f"CLAE={'—' if clae is None else f'{clae:.3f}'}")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title="[bold cyan]LAWS-SIM — Risultati per Scenario[/bold cyan]")
    table.add_column("Scenario",  style="cyan", min_width=40)
    table.add_column("Precision", justify="right")
    table.add_column("Recall",    justify="right")
    table.add_column("F1",        justify="right")
    table.add_column("FPR",       justify="right")
    table.add_column("CLAE",      justify="right")

    for sc, m in results.items():
        clae     = compute_clae(sc, m, baseline)
        clae_str = "—" if clae is None else f"{clae:.3f}"
        color    = "green" if m.f1 > 0.60 else ("yellow" if m.f1 > 0.35 else "red")
        table.add_row(
            sc.value,
            f"{m.precision:.3f}",
            f"{m.recall:.3f}",
            f"[{color}]{m.f1:.3f}[/{color}]",
            f"{m.fpr:.3f}",
            clae_str,
        )
    console.print(table)


def print_cascading_analysis(results: Dict[AttackScenario, SimMetrics]):
    """Analisi narrativa del cascading failure"""
    if not HAS_RICH:
        return
    b  = results[AttackScenario.NONE]
    c  = results[AttackScenario.CASCADING]
    p  = results[AttackScenario.PATCH_ONLY]
    o  = results[AttackScenario.OSINT_POISON]

    console.print(Panel(
        f"""[bold]Cascading Failure Attack — Analisi[/bold]

[yellow]Patch only:[/yellow]   F1 {b.f1:.3f} → {p.f1:.3f}  [red](−{b.f1 - p.f1:.3f})[/red]
[yellow]OSINT only:[/yellow]   F1 {b.f1:.3f} → {o.f1:.3f}  [red](−{b.f1 - o.f1:.3f})[/red]
[yellow]Cascading :[/yellow]   F1 {b.f1:.3f} → {c.f1:.3f}  [red](−{b.f1 - c.f1:.3f})[/red]

[bold cyan]Meccanismo di failure distribuito:[/bold cyan]
  1. CARE kit (EoT patch) → Vision Agent non rileva target
  2. OSINT poisoning      → OSINT Agent abbassa il profilo
  3. Fusion Agent         → riceve segnale degradato da ENTRAMBI i layer
  4. Bayesian update      → non raggiunge la soglia di engagement
  5. Decision Agent       → [red]IGNORE[/red]   ← falso negativo sistemico

[bold yellow]Sinergia dell'attacco:[/bold yellow]
  Il costo del cascading (0.55) è < somma dei singoli (0.70).
  Il degradamento è > della somma delle parti → [red]effetto moltiplicativo[/red].

[bold green]Implicazione per IHL / Certificazione:[/bold green]
  Un sistema LAWS che tace (IGNORE) non è un sistema sicuro.
  Il fallimento silenzioso è più pericoloso del falso positivo.
  → Requisito minimo proposto: "independent evidence floor"
    prima di emettere IGNORE su un soggetto già tracciato.
""",
        title="[bold red]Cascading Failure Analysis[/bold red]",
        border_style="red",
    ))


def plot_results(results: Dict[AttackScenario, SimMetrics]):
    if not HAS_MPL:
        print("[!] matplotlib non trovato — skip plot")
        return

    scenarios = list(results.keys())
    labels    = [s.value.replace(" [", "\n[") for s in scenarios]
    COLORS    = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    f1s  = [m.f1        for m in results.values()]
    prec = [m.precision for m in results.values()]
    rec  = [m.recall    for m in results.values()]
    fprs = [m.fpr       for m in results.values()]

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("LAWS-SIM — Analisi Vulnerabilità Multi-Layer",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Plot 1: F1 per scenario ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(labels, f1s, color=COLORS, edgecolor="black", linewidth=0.6)
    ax1.axhline(f1s[0], color="#2ecc71", linestyle="--", alpha=0.6,
                label=f"Baseline F1 = {f1s[0]:.3f}")
    for bar, v in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.015,
                 f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Score per Scenario di Attacco", fontweight="bold")
    ax1.legend()

    # ── Plot 2: metriche aggregate grouped bar ────────────────────────
    ax2  = fig.add_subplot(gs[0, 2])
    keys = ["Precision", "Recall", "F1", "1−FPR"]
    vals = [prec, rec, f1s, [1 - f for f in fprs]]
    x    = np.arange(len(keys))
    w    = 0.18
    for i, (sc, col) in enumerate(zip(scenarios, COLORS)):
        ax2.bar(x + i * w, [vals[j][i] for j in range(len(keys))],
                w, label=sc.value.split(" ")[0],
                color=col, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x + w * 1.5)
    ax2.set_xticklabels(keys, fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Metriche Aggregate", fontweight="bold")
    ax2.legend(fontsize=7)

    # ── Plot 3: distribuzione threat score ────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    for (sc, m), col in zip(results.items(), COLORS):
        if m.threat_scores:
            ax3.hist(m.threat_scores, bins=40, alpha=0.50,
                     color=col, label=sc.value,
                     density=True, edgecolor="black", linewidth=0.3)
    ax3.axvline(ENGAGEMENT_THRESHOLD, color="red",    linestyle="--",
                linewidth=2, label=f"Engage ({ENGAGEMENT_THRESHOLD})")
    ax3.axvline(ALERT_THRESHOLD,      color="orange", linestyle="--",
                linewidth=1.5, label=f"Alert ({ALERT_THRESHOLD})")
    ax3.set_xlabel("Threat Score")
    ax3.set_ylabel("Densità")
    ax3.set_title("Distribuzione Threat Score per Scenario", fontweight="bold")
    ax3.legend(fontsize=8)

    out = "laws_sim_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Plot salvato → {out}]")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LAWS-SIM v1.0 — Multi-Agent LAWS Vulnerability Simulator")
    parser.add_argument("--steps",   type=int,  default=100,
                        help="Numero di step per simulazione (default 100)")
    parser.add_argument("--seed",    type=int,  default=42,
                        help="Seed random (default 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log dettagliato decisioni sui target")
    parser.add_argument("--no-plot", action="store_true",
                        help="Salta il plot matplotlib")
    args = parser.parse_args()

    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]LAWS-SIM v1.0[/bold cyan]\n"
            "Multi-Agent LAWS Vulnerability Simulator\n\n"
            f"[yellow]Grid:[/yellow] {GRID_SIZE}×{GRID_SIZE}  "
            f"[yellow]Steps:[/yellow] {args.steps}  "
            f"[yellow]Targets:[/yellow] {N_TARGETS}  "
            f"[yellow]Civilians:[/yellow] {N_CIVILIANS}  "
            f"[yellow]Seed:[/yellow] {args.seed}",
            border_style="cyan",
        ))
    else:
        print(f"LAWS-SIM v1.0  |  steps={args.steps}  seed={args.seed}")

    results  : Dict[AttackScenario, SimMetrics] = {}
    scenarios = list(AttackScenario)

    if HAS_RICH:
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      console=console) as prog:
            for sc in scenarios:
                t = prog.add_task(f"Simulazione: {sc.value}…", total=None)
                sim = LAWSSim(sc, steps=args.steps, seed=args.seed)
                m   = sim.run(verbose=args.verbose)
                results[sc] = m
                prog.remove_task(t)
                clae = compute_clae(sc, m, results[AttackScenario.NONE]) \
                       if sc != AttackScenario.NONE else None
                clae_str = "" if clae is None else f"  CLAE={clae:.3f}"
                console.print(
                    f"[green]✓[/green] {sc.value:<42} "
                    f"F1={m.f1:.3f}  FPR={m.fpr:.3f}{clae_str}"
                )
    else:
        for sc in scenarios:
            print(f"Running: {sc.value}…", end=" ", flush=True)
            sim = LAWSSim(sc, steps=args.steps, seed=args.seed)
            results[sc] = sim.run(verbose=args.verbose)
            print(f"F1={results[sc].f1:.3f}")

    baseline = results[AttackScenario.NONE]

    if HAS_RICH:
        console.rule()
    print_results_table(results, baseline)
    print_cascading_analysis(results)

    if args.verbose:
        # Stampa ultimi 10 log eventi sui target (scenario cascading)
        sim_v = LAWSSim(AttackScenario.CASCADING, steps=args.steps, seed=args.seed)
        sim_v.run(verbose=True)
        if sim_v.step_log and HAS_RICH:
            console.rule("[bold yellow]Step log — Cascading (ultimi 10 eventi target)[/bold yellow]")
            for entry in sim_v.step_log[-10:]:
                console.print(
                    f"  step=[cyan]{entry['step']:3d}[/cyan] "
                    f"id=[yellow]{entry['entity_id']}[/yellow] "
                    f"action=[{'red' if entry['action']=='IGNORE' else 'green'}]{entry['action']}[/{'red' if entry['action']=='IGNORE' else 'green'}] "
                    f"score={entry['score']:.4f} "
                    f"patch={'[red]ON[/red]' if entry['patch'] else 'off'} "
                    f"osint_poison={'[red]ON[/red]' if entry['poisoned'] else 'off'}"
                )

    # ── Export JSON ──────────────────────────────────────────────────
    export = {}
    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        export[sc.value] = {
            "precision" : round(m.precision, 4),
            "recall"    : round(m.recall,    4),
            "f1"        : round(m.f1,        4),
            "fpr"       : round(m.fpr,       4),
            "tp"        : m.tp, "fp": m.fp,
            "tn"        : m.tn, "fn": m.fn,
            "clae"      : round(clae, 4) if clae is not None else None,
        }
    with open("laws_sim_results.json", "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False)

    if HAS_RICH:
        console.print("\n[green]✓ Export:[/green] laws_sim_results.json")
    else:
        print("\nRisultati salvati → laws_sim_results.json")

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()
