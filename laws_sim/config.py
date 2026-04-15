"""
LAWS-SIM — Configuration constants with scientific grounding.

References:
- YOLOv8 outdoor confidence 99.1%: Sodhro, A.H. et al. (2025). "Real-time efficiency 
  of YOLOv5 and YOLOv8 in human intrusion detection across diverse environments". 
  Internet of Things, 33, 101707. https://doi.org/10.1016/j.iot.2025.101707
- Threod EOS C VTOL cruise speed 61 km/h: Threod Systems (2025). 
  https://www.threod.com/eos-c-vtol/
- RemoEye-002B max speed 75 km/h: Uconsystem. https://tradekorea.com/product/detail/P580428/RemoEye-002B.html
- HRVS radar altimeter (<300 m): AIR TEAM. https://www.airteam.eu/fr/p/radar-velocity-system-for-uav-350m-altitude
- GEN-TPRM OSINT risk model: Asili, H. & Bahtiyar, Ş. (2025). 
  https://doi.org/10.1109/ACIT65614.2025.11185738
- ETIP threat scoring: González-Granadillo, G. et al. (2021). 
  https://doi.org/10.1016/j.jisa.2020.102658
"""

# =============================================================================
# SIMULATION ENVIRONMENT
# =============================================================================
GRID_SIZE = 30                # simulation world size (arbitrary units)
N_TARGETS = 3                 # number of threat targets
N_CIVILIANS = 15              # number of civilians

# Drone movement (realistic tactical UAS parameters)
DRONE_CRUISE_SPEED_KMH = 61
DRONE_MAX_SPEED_KMH = 90
DRONE_ALTITUDE_M = 10         # FIX: era 300, che rendeva dist >= 300 > YOLO_MAX_RANGE=150
                               # → nessuna entità mai visibile → tutti zero.
                               # 10m è l'altitudine operativa tipica per detection ravvicinata.
DRONE_STEP_SIZE = 2

# Detection range
YOLO_MIN_RANGE = 15.0
YOLO_MAX_RANGE = 150.0        # Con altitudine 10m, la distanza 3D è sqrt(x²+y²+100)
                               # che scala correttamente col raggio 150m.

# =============================================================================
# FUSION & DECISION THRESHOLDS
# =============================================================================
FUSION_WEIGHTS = {
    "vision": 0.45,
    "osint": 0.35,
    "behavioral": 0.20
}

ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD = 0.38
TRACK_THRESHOLD = 0.22

# =============================================================================
# VISION AGENT (YOLOv8)
# =============================================================================
BASELINE_CONFIDENCE = 0.991   # Sodhro et al., 2025
DETECTION_THRESHOLD = 0.50

# Parametri modello analitico (fallback senza YOLO reale)
PATCH_SUPPRESSION = 0.65
PATCH_DIST_FALLOFF = 0.04

# =============================================================================
# OSINT PROFILING
# =============================================================================
OSINT_FIELDS_TOTAL = 10
OSINT_FIELDS_POISONED = 3

# =============================================================================
# ADVERSARIAL PATCH OPTIMIZATION (EoT)
# =============================================================================
PATCH_H = 100                 # FIX: era 240 → patch enorme che copriva tutta la faccia.
PATCH_W = 80                  # 100×80 px su immagine 640×640 = ~2.5% dell'immagine,
                               # realistico per un logo stampato su indumento , tipo A5, boh?!
PATCH_LR = 0.03
PATCH_STEPS = 80
PATCH_EPS = 0.05
EOT_N_TRANSFORMS = 8
PERSON_CLASS_ID = 0           # COCO classe 0 = "person"
IMG_SIZE = 640                # input size YOLOv8

# C_vision per CLAE: patch_area / bbox_area (aggiornato con nuove dimensioni)
# Su un bbox medio di ~200×400px: (100×80)/(200×400) = 0.10
PATCH_BBOX_COVERAGE = 0.10