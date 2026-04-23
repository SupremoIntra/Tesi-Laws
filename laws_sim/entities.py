"""
Gestione entità simulate -> civilians, targets, OSINT profiles e environment.
"""

import random
import math
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

#importo costanti da config.py (paper)
from config import (
    GRID_SIZE, N_TARGETS, N_CIVILIANS, DRONE_ALTITUDE_M, DRONE_STEP_SIZE,
    YOLO_MAX_RANGE
)

# Genero dati sintetici come nazionalità, nome ecc. (mi baso però sui punteggi di rischio)
try:
    from faker import Faker
    _fake = Faker(["it_IT", "en_US"])
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False


class AgentRole(Enum):
    CIVILIAN = "civilian"
    TARGET = "target"


@dataclass
class OSINTProfile:
    """
    Costruzione profilo di un'entità -> indicatori sintetici
    --> da aggiungere paper...
    
    Inspired by:
    - GEN-TPRM: Asili & Bahtiyar (2025) — OSINT-driven risk assessment
      https://doi.org/10.1109/ACIT65614.2025.11185738
    - ETIP: González-Granadillo et al. (2021) — heuristic threat scoring https://doi.org/10.1016/j.jisa.2020.102715
    """
    name: str
    age: int
    nationality: str
    social_score: float
    geo_anomaly: float
    network_centrality: float
    is_poisoned: bool = False

    @property
    def threat_score(self) -> float:
        """Weighted threat score presi dagli indicatori OSINT."""
        return float(np.clip(
            0.40 * self.social_score +
            0.35 * self.geo_anomaly +
            0.25 * self.network_centrality, 0, 1))


class SimEntity:
    """Entita siulata nell'ambiente."""

    def __init__(self, eid: int, role: AgentRole, grid_size: int):
        self.id = eid
        self.role = role
        self.grid = grid_size
        self.x = random.randint(0, grid_size - 1)
        self.y = random.randint(0, grid_size - 1)
        self.history: List[Tuple[int, int]] = [(self.x, self.y)]
        self.care_kit_active = False
        self.osint_profile = self._gen_profile()

    def _gen_profile(self) -> OSINTProfile:
        """generazione dati sintetici basato sul faker di prima (opzionale) """
        name = _fake.name() if HAS_FAKER else f"Entity_{self.id:03d}"
        nat = _fake.country() if HAS_FAKER else "N/A"
        if self.role == AgentRole.TARGET:
            return OSINTProfile(
                name=name,
                age=random.randint(20, 45),
                nationality=nat,
                social_score=random.uniform(0.55, 0.90),
                geo_anomaly=random.uniform(0.50, 0.85),
                network_centrality=random.uniform(0.40, 0.80)
            )
        return OSINTProfile(
            name=name,
            age=random.randint(18, 70),
            nationality=nat,
            social_score=random.uniform(0.00, 0.30),
            geo_anomaly=random.uniform(0.00, 0.25),
            network_centrality=random.uniform(0.00, 0.20)
        )

    def move(self) -> None:
        """muovo le entità in modo casuale ma realistico (target più veloci)"""
        step = 2 if self.role == AgentRole.TARGET else 1
        self.x = int(np.clip(self.x + random.randint(-step, step), 0, self.grid - 1))
        self.y = int(np.clip(self.y + random.randint(-step, step), 0, self.grid - 1))
        self.history.append((self.x, self.y))
        if len(self.history) > 20:
            self.history.pop(0)

    @property
    def behavioral_score(self) -> float:
        """Lo score di "anomalia" basato sui movimenti recenti (varianza) -> più è alto, più è sospetto"""
        if len(self.history) < 3:
            return 0.0
        arr = np.array(self.history[-10:])
        score = np.clip((np.var(arr[:, 0]) + np.var(arr[:, 1])) / 20.0, 0, 1)
        if self.role == AgentRole.TARGET:
            score = float(np.clip(score + random.uniform(0.10, 0.25), 0, 1))
        return float(score)


class Environment:
    """L'enviroment vero e proprio con droni e entità"""

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid = grid_size
        self.step_idx = 0
        self.drone_x = grid_size // 2
        self.drone_y = grid_size // 2
        self.entities: List[SimEntity] = []

        # inizializzo targets e civilians
        for i in range(N_TARGETS):
            self.entities.append(SimEntity(i, AgentRole.TARGET, grid_size))
        for i in range(N_CIVILIANS):
            self.entities.append(SimEntity(N_TARGETS + i, AgentRole.CIVILIAN, grid_size))

    def tick(self) -> None:
        """Faccio un passo di simulazione (tick)"""
        for e in self.entities:
            e.move()
        # drone si muove a velocità di crociera (fino a DRONE_STEP_SIZE unità per tick)
        self.drone_x = int(np.clip(self.drone_x + random.randint(-DRONE_STEP_SIZE, DRONE_STEP_SIZE), 0, self.grid - 1))
        self.drone_y = int(np.clip(self.drone_y + random.randint(-DRONE_STEP_SIZE, DRONE_STEP_SIZE), 0, self.grid - 1))
        self.step_idx += 1

    def dist_to_drone(self, entity: SimEntity) -> float:
        """Distanza tra drone e entità (dist 3D considerando altitudine)"""
        return math.sqrt(
            (entity.x - self.drone_x) ** 2 +
            (entity.y - self.drone_y) ** 2 +
            DRONE_ALTITUDE_M ** 2
        )

    def visible(self) -> List[SimEntity]:
        """Mi da le entità che si trovano nel range di rilevamento del drone (YOLO_MAX_RANGE)"""
        return [e for e in self.entities if self.dist_to_drone(e) <= YOLO_MAX_RANGE]

    def nearby_civilians(self, target: SimEntity, radius: int = 5) -> int:
        """Conto i civilians vicini a un target entro un certo raggio (per valutare rischi collaterali)"""
        return sum(
            1 for e in self.entities
            if e.role == AgentRole.CIVILIAN
            and abs(e.x - target.x) + abs(e.y - target.y) <= radius
        )