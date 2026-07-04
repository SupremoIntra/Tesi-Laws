"""
Gestione entità simulate -> civilians, targets, OSINT profiles e environment.
-- v1 con faker random, nomi finti
-- v2 generazione sintetica data-driven (dsitrib statistiche reali per modellare metadati OSINT --> poi ci applico inferenza bayesiana esatta)
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

class AgentRole(Enum):
    CIVILIAN = "civilian"
    TARGET = "target"


@dataclass
class OSINTProfile:
    """
    Rete Bayesiana nativa per OSINT --> simula una sorta di OSINT (parte Buffa)
    Modella il profilo campionando i dati da distribuzioni statistiche formali
    calibrate per simulare database reali (es. ACLED Armed Conflict Location & Event Data per geo-rischio -- maggiore in zone di guerra, liste OFAC - targa in blacklist governativa).
    """
    def __init__(self, role: AgentRole):
        self.is_target = (role == AgentRole.TARGET)
        self.is_poisoned = False
        
        if self.is_target:
            # DISTRIBUZIONE DI BERNOULLI: Modella un evento binario (Sì/No). 
            # I target hanno un'alta probabilità (75%) di essere già in una blacklist di targhe.
            self.plate_blacklist = random.random() < 0.75  
            
            # DISTRIBUZIONE BETA: Crea una curva di probabilità limitata tra 0 e 1.
            # (alpha=8, beta=2) sbilancia la curva verso valori alti (es. 0.7-0.9), simulando
            # la presenza in una zona ad alto rischio di conflitto.
            self.geo_risk = random.betavariate(8, 2) 
            
            # DISTRIBUZIONE DI POISSON: Modella il conteggio di eventi discreti nel tempo.
            # lam=12 significa che ci aspettiamo in media 12 tracce social/menzioni sospette.
            self.social_matches = np.random.poisson(lam=12) 
        else:
            # Civili innocenti: i parametri sono invertiti.
            # Tasso di falso positivo bassissimo (1%) per la blacklist.
            self.plate_blacklist = random.random() < 0.01  
            # Curva Beta (alpha=2, beta=8) sbilanciata verso valori bassi (zona sicura).
            self.geo_risk = random.betavariate(2, 8)       
            # Poisson con media 1: pochissime menzioni rilevanti.
            self.social_matches = np.random.poisson(lam=1)

    @property
    def threat_score(self) -> float:
        """
        calcolo P(Minaccia | Evidenze OSINT) usando bayes.
        Sostituisce la precedente media pesata con un'inferenza rigorosa.
        """
        # Probabilità a priori (assumiamo che il 15% delle entità nell'area sia ostile) --> assunzione euristica
        p_threat_prior = 0.15
        p_civil_prior = 0.85
        
        # Calcolo delle Likelihood (Verosimiglianze) per un Target
        lh_b_threat = 0.75 if self.plate_blacklist else 0.25
        lh_g_threat = max(self.geo_risk, 0.01)
        lh_s_threat = max(1.0 - math.exp(-0.3 * self.social_matches), 0.01)
        
        # Calcolo delle Likelihood per un Civile Innocente
        lh_b_civil = 0.01 if self.plate_blacklist else 0.99
        lh_g_civil = max(1.0 - self.geo_risk, 0.01)
        lh_s_civil = max(math.exp(-0.6 * self.social_matches), 0.01)
        
        # Inferenza Bayesiana: Numeratori e normalizzazione (Evidenza totale)
        num_threat = p_threat_prior * lh_b_threat * lh_g_threat * lh_s_threat
        num_civil = p_civil_prior * lh_b_civil * lh_g_civil * lh_s_civil
        
        evidence = num_threat + num_civil
        
        # Ritorna il Threat Score finale [0, 1]
        return round(num_threat / evidence, 4) if evidence > 0 else 0.0


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
        self.osint_profile = OSINTProfile(self.role)


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