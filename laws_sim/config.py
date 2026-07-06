"""
Configurazione LAWS-SIM — Ottimizzata per Apple Silicon M4 16GB (2026)

Questa configurazione bilancia:
- Rigore accademico (loss asintotica, EoT completo)
- Stabilità hardware (gradient accumulation, Float32 per MPS safety)
- Tempi di training accettabili (~45-60 min per 1500 step)

Ogni costante di dominio (velocità, altitudine, soglie etiche, pesi OSINT)
è derivata da una fonte esplicita elencata sotto, per garantire riproducibilità
scientifica in sede di discussione della tesi.

Fonti citate:
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
         Riferimento per il Principio di Distinzione IHL in scenari di
         Asymmetric Warfare / Urban Clutter.
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

# === Training Patch ===
PATCH_LR = 0.01  # Learning Rate ridotto per evitare saturazione sigmoide [9]
PATCH_STEPS = 1500  # Step totali (ridotto per evitare overfitting) [9]

# EoT (Expectation over Transformation) [10]
EOT_N_TRANSFORMS = 16  # Numero di trasformazioni per immagine

# === Ottimizzazione Hardware (M4 16GB) ===
BATCH_SIZE_PHYSICAL = 1  # Batch fisico per evitare OOM su MPS
GRADIENT_ACCUMULATION_STEPS = 4  # Batch effettivo = 1 * 4 = 4
TV_WEIGHT = 0.1  # Pattern low-freq fisicamente stampabili [8]

# === Checkpoint e Early Stopping ===
CHECKPOINT_EVERY = 100
EARLY_STOPPING_PATIENCE = 200
EARLY_STOPPING_WINDOW = 50

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

# === Popolazione ambiente simulato (Asymmetric Warfare / Urban Clutter) ===
# Rapporto 1:10 target/civili deliberatamente sbilanciato [11]: modella uno
# scenario di "Urban Clutter" in cui il Principio di Distinzione IHL è messo
# sotto stress massimo. Un rapporto realistico e sfavorevole è necessario per
# dimostrare che, sotto Data Poisoning OSINT (che eleva artificialmente il
# profilo di rischio dei civili), il Fusion Agent Bayesiano collassa verso
# violazioni del Principio di Proporzionalità (falsi positivi su civili),
# non per sotto-calibrazione casuale ma per pressione strutturale del contesto.
N_TARGETS = 3
N_CIVILIANS = 30

GRID_SIZE = 100  # Dimensione griglia simulatore (metri)

# === Adversarial Patch — coverage fisico ===
# 0.20 = 0.5 (larghezza) * 0.4 (altezza) derivato direttamente da
# get_chest_bbox_proportional ("Tactical Vest Assumption"): un giubbotto
# tattico/plate carrier sul petto copre ~20% della sagoma intera vista
# dall'alto. Valore ora coerente al 100% con la logica di iniezione reale
# usata sia in training (patch_optimizer.py) sia in evaluate_on_dataset.
PATCH_BBOX_COVERAGE = 0.20

# === Fusion Weights — pesi del Bayesian Sensor Fusion ===
# Vision 45%: sensore primario (YOLOv8, [1] conferma 99.1% confidence baseline
#   outdoor in condizioni nominali → merita peso maggiore ma non esclusivo).
# OSINT 35%: rischio da metadati esterni (blacklist targa, geo-rischio, social
#   match), calibrato secondo GEN-TPRM [5] ed ETIP [6].
# Behavioral 20%: peso minimo intenzionale — un sistema che pesa troppo "come
#   ti muovi" rispetto a "cosa stai facendo ora" rischia di giudicare
#   l'individuo sul profilo anziché sull'azione presente.
FUSION_WEIGHTS = {"vision": 0.45, "osint": 0.35, "behavioral": 0.20}

# === Soglie decisionali (Diritto Internazionale Umanitario) ===
# ENGAGE (0.58): soglia di ingaggio letale. Deliberatamente vicina al 50%
#   per riflettere, criticamente, l'abbassamento degli standard di certezza
#   operativa osservato in teatri di guerra reali (es. sistema "Lavender"),
#   dove pressione tattica e Automation Bias spingono ad autorizzare
#   attacchi con confidenza appena superiore al lancio di una moneta.
#   Questo NON è un endorsement — è il punto critico della tesi: mostrare
#   quanto sia fragile un sistema calibrato su questi standard.
ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD = 0.38    # Stato di allerta, revisione umana richiesta
TRACK_THRESHOLD = 0.22    # Soglia minima per iniziare il tracking passivo

# === OSINT Data Poisoning ===
# I 3 campi del profilo (plate_blacklist, geo_risk, social_matches) sono
# TUTTI alterabili dall'attacco simulato in simulator.py → poisoning totale.
OSINT_FIELDS_TOTAL = 3
OSINT_FIELDS_POISONED = 3

# NOTA ARCHITETTURALE: il modello analitico euristico (PATCH_SUPPRESSION /
# PATCH_DIST_FALLOFF) è stato deliberatamente ESCLUSO da questa versione.
# Decisione motivata: un modello a decadimento euristico basato sulla sola
# distanza non ha fondamento nei pesi convoluzionali reali della rete e
# inquinerebbe il Threat Model con un secondo canale di rumore non
# tracciabile. Il degrado visivo è modellato ESCLUSIVAMENTE in modo
# data-driven, leggendo l'Evasion Rate empirico misurato offline su YOLOv8
# via il bridge vision_metrics.json (vedi detection.py).

# === File e Output (percorsi assoluti dentro outputs/) ===
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_DIR, "checkpoint_patch.pt")
BEST_PATCH_FILE = os.path.join(PATCHES_DIR, "care_kit_patch_universal.pt")
METRICS_JSON_FILE = os.path.join(METRICS_DIR, "training_metrics.json")
VISION_METRICS_JSON = os.path.join(METRICS_DIR, "vision_metrics.json")
