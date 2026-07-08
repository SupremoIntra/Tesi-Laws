"""
Configurazione LAWS-SIM.

Costanti derivate dai paper citati sotto, usate sia dal training della
patch (visione) sia dal simulatore multi-agente.

Fonti:
    [1] Sodhro, A.H. et al. (2025). "Real-time efficiency of YOLOv5 and YOLOv8
        in human intrusion detection across diverse environments".
        Internet of Things, 33, 101707. https://doi.org/10.1016/j.iot.2025.101707
    [2] Threod Systems (2025). EOS-C VTOL, cruise speed 61 km/h.
        https://www.threod.com/eos-c-vtol/
    [3] Uconsystem. RemoEye-002B, max speed 75 km/h.
        https://tradekorea.com/product/detail/P580428/RemoEye-002B.html
    [4] AIR TEAM. HRVS radar altimeter, range < 300/350 m.
        https://www.airteam.eu/fr/p/radar-velocity-system-for-uav-350m-altitude
    [5] Asili, H. & Bahtiyar, Ş. (2025). GEN-TPRM OSINT risk model.
        https://doi.org/10.1109/ACIT65614.2025.11185738
    [6] González-Granadillo, G. et al. (2021). ETIP threat scoring.
        https://doi.org/10.1016/j.jisa.2020.102658
    [7] Carlini, N. & Wagner, D. (2017). "Towards Evaluating the Robustness
        of Neural Networks".
    [8] Thys, S. et al. (2019). "Fooling automated surveillance cameras".
    [9] Brown, T.B. et al. (2017). "Adversarial Patch".
    [10] Athalye, A. et al. (2017). "Synthesizing Robust Adversarial Examples".
    [11] Arkin, R.C. (2009). "Governing Lethal Behavior in Autonomous Robots".
"""

import os

# === Percorsi base (relativi alla root del progetto) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
PATCHES_DIR = os.path.join(OUTPUTS_DIR, "patches")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")

# === Visione (Patch Optimizer — training) ===
IMG_SIZE = 640
PATCH_H = 200  # Altezza patch base (spazio logits, training)
PATCH_W = 200  # Larghezza patch base (spazio logits, training)
PERSON_CLASS_ID = 0  # Classe "person" in COCO/VisDrone
LOSS_TOP_K = 20  # Media sulle top-K celle piu' confidenti invece che su tutta la mask (Thys et al. 2019, max objectness)

# === Training Patch ===
PATCH_LR = 0.01  # Learning Rate ridotto per evitare saturazione sigmoide [9]
PATCH_STEPS = 5000  # Fase 3 (top-k loss): meta' del budget di Fase 1, pensato per stare in ~4 ore

# EoT (Expectation over Transformation) [10]
EOT_N_TRANSFORMS = 16  # Numero di trasformazioni per immagine

# === Ottimizzazione Hardware (M4 16GB) ===
BATCH_SIZE_PHYSICAL = 1  # Batch fisico per evitare OOM su MPS
GRADIENT_ACCUMULATION_STEPS = 4  # Torna a 4 (16 e' risultato peggiore, vedi thesis_notes.md) -> 1250 update reali
TV_WEIGHT = 0.1  # Pattern low-freq fisicamente stampabili [8]

# === Checkpoint e Early Stopping ===
CHECKPOINT_EVERY = 100
EARLY_STOPPING_PATIENCE = 1000  # Scalato per 1250 update potenziali (5000/4)
EARLY_STOPPING_WINDOW = 100

# === Simulation Environment (Digital Twin del drone tattico) ===
# Altitudine operativa bassa per identificazione ravvicinata (non ricognizione
# d'area): coerente con droni VTOL tattici [2][3], non con un HRVS da 300m [4].
DRONE_ALTITUDE_M = 10

# YOLO_MAX_RANGE: raggio massimo (in metri simulati) entro cui il sensore
# genera detection. Con GRID_SIZE=100 e altitudine 10m, la diagonale massima
# del campo è sqrt(100^2+100^2+10^2) ≈ 141.8m → 150m garantisce piena
# coverage del campo simulato senza saturare il range fisico dell'HRVS [4].
YOLO_MAX_RANGE = 150.0

# Velocità di crociera derivata dal Threod EOS-C VTOL [2]: 61 km/h.
# Conversione in step di griglia (assunzione: 1 tick ≈ 1s, 1 unità griglia ≈ 1m).
DRONE_SPEED_KMH = 61.0
DRONE_STEP_SIZE = round(DRONE_SPEED_KMH * 1000 / 3600)  # ≈ 17 m/tick

DETECTION_THRESHOLD = 0.50  # Soglia confidenza per detection

# === Popolazione ambiente simulato ===
# Rapporto 1:10 target/civili (Urban Clutter, Asymmetric Warfare [11]).
# Sotto Data Poisoning OSINT, questo sbilanciamento stressa il Fusion
# Agent verso falsi positivi sui civili.
N_TARGETS = 3
N_CIVILIANS = 30

GRID_SIZE = 100  # Dimensione griglia simulatore (metri)

# === Adversarial Patch — coverage fisico ===
# 0.5 (larghezza) * 0.4 (altezza) = 0.20, stesso valore usato in
# get_chest_bbox_proportional. Deve restare coerente con quella funzione.
PATCH_BBOX_COVERAGE = 0.20

# === Fusion Weights — pesi del Bayesian Sensor Fusion ===
# Vision 45% (YOLOv8, baseline 99.1% outdoor [1]), OSINT 35%
# (GEN-TPRM [5], ETIP [6]), Behavioral 20%.
FUSION_WEIGHTS = {"vision": 0.45, "osint": 0.35, "behavioral": 0.20}

# === Soglie decisionali ===
# ENGAGE vicina al 50% per modellare l'abbassamento degli standard di
# certezza operativa osservato in teatri di guerra reali (Automation
# Bias, es. sistema "Lavender").
ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD = 0.38    # Stato di allerta, revisione umana richiesta
TRACK_THRESHOLD = 0.22    # Soglia minima per iniziare il tracking passivo

# === OSINT Data Poisoning ===
# I 3 campi del profilo (plate_blacklist, geo_risk, social_matches) sono
# tutti alterabili dall'attacco simulato in simulator.py.
OSINT_FIELDS_TOTAL = 3
OSINT_FIELDS_POISONED = 3

# Nessun modello analitico di decadimento distanza-based: il degrado
# visivo è solo data-driven, da vision_metrics.json (vedi detection.py).

# === File e Output (percorsi assoluti dentro outputs/) ===
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_DIR, "checkpoint_patch.pt")
BEST_PATCH_FILE = os.path.join(PATCHES_DIR, "care_kit_patch_universal.pt")
METRICS_JSON_FILE = os.path.join(METRICS_DIR, "training_metrics.json")
VISION_METRICS_JSON = os.path.join(METRICS_DIR, "vision_metrics.json")
