"""
LAWS-SIM v3.0 — Configuration constants with scientific grounding.

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
DRONE_CRUISE_SPEED_KMH = 61   # Threod EOS C VTOL cruise speed (https://www.threod.com/eos-c-vtol/)
DRONE_MAX_SPEED_KMH = 90      # Threod EOS C VTOL max speed
DRONE_ALTITUDE_M = 300        # typical operational altitude (HRVS radar tested up to 300m)
DRONE_STEP_SIZE = 2           # discretized movement step (derived from speed / simulation tick)

# Detection range constraints
YOLO_MIN_RANGE = 15.0         # minimum detection range (close proximity)
YOLO_MAX_RANGE = 150.0        # maximum detection range (sensor limits)

# =============================================================================
# FUSION & DECISION THRESHOLDS
# =============================================================================
FUSION_WEIGHTS = {
    "vision": 0.45,
    "osint": 0.35,
    "behavioral": 0.20
}

# Decision thresholds (calibrated to balance sensitivity and specificity)
ENGAGEMENT_THRESHOLD = 0.58
ALERT_THRESHOLD = 0.38
TRACK_THRESHOLD = 0.22

# =============================================================================
# VISION AGENT (YOLOv8)
# =============================================================================
# Baseline confidence from real-world outdoor testing: 99.1% (Sodhro et al., 2025)
BASELINE_CONFIDENCE = 0.991
DETECTION_THRESHOLD = 0.50   # standard YOLO confidence threshold

# Analytical model fallback parameters
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
PATCH_H = 100
PATCH_W = 80
PATCH_LR = 0.03
PATCH_STEPS = 80
PATCH_EPS = 0.05
EOT_N_TRANSFORMS = 8        # number of EoT transforms per optimization step
PERSON_CLASS_ID = 0         # COCO class ID for "person"
IMG_SIZE = 640              # YOLOv8 input size

# Physical coverage ratio (patch area / bounding box area)
PATCH_BBOX_COVERAGE = 0.28  # realistic for a wearable adversarial pattern