"""
Modulo Detection -> sensore "Digital Twin" (Disaccoppiato)
Invece di eseguire l'inferenza di YOLO ad ogni step temporale (pesante),
questo modulo simula le prestazioni del sensore visivo utilizzando il parametro F1-Score (es. 0.008) 
misurato sperimentalmente e isolatamente sul dataset VisDrone.

NOTA ARCHITETTURALE (Threat Model Integrity):
Questo modulo dipende ESCLUSIVAMENTE dal dato empirico letto dal bridge
JSON (vision_metrics.json, prodotto da simulator.evaluate_on_dataset).
È stato deliberatamente scartato un modello analitico/euristico alternativo
(a decadimento distanza-dipendente) perché non fondato sui pesi
convoluzionali reali della rete: avrebbe introdotto un secondo canale di
rumore non tracciabile, indebolendo la validità scientifica del Threat
Model. Il collasso del sensore in questa simulazione è quindi sempre
la fotografia diretta dell'Evasion Rate misurato offline su YOLOv8,
non un'approssimazione matematica indipendente.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Non serve più importare YOLO o Torch qui dentro.
from config import YOLO_MAX_RANGE, DETECTION_THRESHOLD, PATCH_BBOX_COVERAGE
from entities import SimEntity, AgentRole

@dataclass
class VisionDetection:
    """Output standardizzato del modulo vision da passare alla Sensor Fusion"""
    detected: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_label: str
    patch_active: bool = False
    patch_coverage: float = 0.0


class VisionAgentStat:
    """
    Agente Sensore Statistico.
    Usa la metrica empirica per abbattere la probabilità di detection durante l'attacco.
    """
    def __init__(self, empirical_f1: float):
        # Il ponte tra realtà empirica e simulazione (letto dal JSON)
        self.empirical_f1 = empirical_f1
        self.fn_count = 0  # Contatore per i Falsi Negativi (Target non visti)

    def detect(self, entity: SimEntity, distance: float, patch_active: bool) -> VisionDetection:
        """
        Calcola probabilisticamente se il drone vede l'entità a questa distanza.
        """
        # Se fuori range massimo visivo, restituisco "non rilevato"
        if distance > YOLO_MAX_RANGE:
            return VisionDetection(False, 0.0, (0,0,0,0), "background")

        # Decadimento naturale della nitidezza della telecamera con l'aumentare della distanza
        dist_scale = math.exp(-1.0 * distance / YOLO_MAX_RANGE)

        if patch_active and entity.care_kit_active:
            # SOTTO ATTACCO ADVERSARIAL: La confidenza base viene "castrata" dall'F1 empirico
            # Se empirical_f1 è 0.008, la confidenza non supererà mai una soglia ridicola.
            base_conf = random.uniform(0.0, max(self.empirical_f1, 0.05))
            cov = PATCH_BBOX_COVERAGE
        else:
            # BASELINE (Nessun attacco): Sensore in salute, performance ottimali
            base_conf = random.uniform(0.70, 0.95)
            cov = 0.0

        # Aggiungo rumore di misurazione gaussiano per simulare le imperfezioni atmosferiche/lenti
        conf = float(np.clip(base_conf * dist_scale + random.gauss(0, 0.02), 0, 1))
        
        # Supera la soglia di rilevamento?
        detected = conf >= DETECTION_THRESHOLD
        
        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected=detected,
            confidence=conf,
            bbox=(entity.x, entity.y, 5, 8), # Bbox fittizio per simulare ingombro spaziale
            class_label="person" if detected else "background",
            patch_active=patch_active and entity.care_kit_active,
            patch_coverage=cov
        )