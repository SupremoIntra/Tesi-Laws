"""
Modulo Detection -> sensore "Digital Twin" (Disaccoppiato)

Invece di eseguire l'inferenza di YOLO ad ogni step temporale (pesante),
questo modulo simula le prestazioni del sensore visivo come TEST DI BERNOULLI
sui recall empirici misurati sul dataset (R1 = sensitivita', R2 = specificita'),
letti da vision_metrics.json.

REVISIONE (fix D1, D3, D4):
  - D1: la detection non e' piu' un cap deterministico su un F1 aggregato.
        F1 e' una media armonica di precision e recall: usarla come tetto
        della confidenza di un singolo frame non ha interpretazione
        probabilistica valida. La quantita' corretta per "il target viene
        visto?" e' R1, che e' esattamente un tasso di successo per frame
        positivo. Ora: detected ~ Bernoulli(R1).
  - D3: i civili non cadono piu' sempre nel ramo "sensore in salute".
        La loro probabilita' di falso allarme e' 1 - R2, quindi la
        specificita' misurata entra finalmente nel simulatore.
  - D4: il decadimento con la distanza NON entra piu' nella decisione.
        R1 e' una media empirica misurata su frame reali che contengono gia'
        una distribuzione di distanze eterogenee: applicare di nuovo un
        decadimento analitico conterebbe l'effetto due volte. dist_scale
        resta solo come modulatore della confidenza riportata nei log.
        -> Da dichiarare in tesi come scelta di modellazione esplicita.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Non serve importare YOLO o Torch qui dentro.
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

    Modella la detection come estrazione di Bernoulli con probabilita' pari
    al recall empirico appropriato:
      - target  -> R1 (post-attacco se la patch e' attiva su quel target)
      - civile  -> 1 - R2 (tasso di falso allarme)

    Nota su R2: e' invariante per costruzione nel disegno sperimentale
    attuale (la patch non viene disegnata nei frame negativi), quindi
    r2_pre == r2_post. La struttura esplicita rende il codice corretto a
    prescindere: se il disegno cambiasse, funzionerebbe senza modifiche.
    """

    def __init__(
        self,
        r1_pre: float,
        r1_post: float,
        r2_pre: float,
        r2_post: float,
    ) -> None:
        """
        Args:
            r1_pre:  sensitivita' senza attacco (baseline reale del dataset)
            r1_post: sensitivita' sotto attacco patch
            r2_pre:  specificita' senza attacco
            r2_post: specificita' sotto attacco
        """
        # ATTENZIONE: firma cambiata rispetto alla versione precedente
        # (era empirical_f1 singolo). Verificare tutti i punti di
        # istanziazione -> attualmente solo simulator.LAWSSim.__init__.
        self.r1_pre = float(np.clip(r1_pre, 0.0, 1.0))
        self.r1_post = float(np.clip(r1_post, 0.0, 1.0))
        self.r2_pre = float(np.clip(r2_pre, 0.0, 1.0))
        self.r2_post = float(np.clip(r2_post, 0.0, 1.0))
        self.fn_count = 0  # Falsi Negativi (target non visti)

    def detect(
        self,
        entity: SimEntity,
        distance: float,
        patch_active: bool,
    ) -> VisionDetection:
        """
        Determina probabilisticamente se il drone vede l'entita'.

        La decisione dipende SOLO dal recall empirico (test di Bernoulli).
        Distanza e rumore gaussiano modulano la confidenza riportata, non
        l'esito: vedi nota D4 nel docstring del modulo.
        """
        if distance > YOLO_MAX_RANGE:
            return VisionDetection(False, 0.0, (0, 0, 0, 0), "background")

        # Solo estetico/di log: NON entra in `detected`.
        dist_scale = math.exp(-1.0 * distance / YOLO_MAX_RANGE)

        patch_on_target = bool(patch_active and entity.care_kit_active)

        if entity.role == AgentRole.TARGET:
            # Frame positivo: probabilita' di essere visto = sensitivita'
            p_detect = self.r1_post if patch_on_target else self.r1_pre
            cov = PATCH_BBOX_COVERAGE if patch_on_target else 0.0
        else:
            # Civile = frame negativo: probabilita' di falso allarme = 1 - R2
            p_detect = 1.0 - (self.r2_post if patch_active else self.r2_pre)
            cov = 0.0

        detected = random.random() < p_detect

        # Confidenza coerente con l'esito, poi modulata da distanza e rumore
        # (imperfezioni atmosferiche/lenti). Il clip garantisce [0,1].
        conf_raw = (
            random.uniform(DETECTION_THRESHOLD, 0.95) if detected
            else random.uniform(0.05, DETECTION_THRESHOLD)
        )
        conf = float(np.clip(conf_raw * dist_scale + random.gauss(0, 0.02), 0.0, 1.0))

        if not detected and entity.role == AgentRole.TARGET:
            self.fn_count += 1

        return VisionDetection(
            detected=detected,
            confidence=conf,
            bbox=(entity.x, entity.y, 5, 8),  # ingombro spaziale fittizio
            class_label="person" if detected else "background",
            patch_active=patch_on_target,
            patch_coverage=cov,
        )
