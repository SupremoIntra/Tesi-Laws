# LAWS-SIM v3.0

Simulatore multi‑agente di un Lethal Autonomous Weapons System (LAWS) con YOLOv8 reale, attacco adversarial patch (EoT) e metrica proposta CLAE.

## Installazione
```python
pip install torch torchvision ultralytics opencv-python rich matplotlib faker
```
## Comandi principali

### Simulazione base (modello analitico)

```python
python -m laws_sim.cli --steps 150
```

### Simulazione con YOLO reale

```python
python -m laws_sim.cli --real-yolo --steps 200
```

Se disponi di un dataset di immagini (es. VisDrone):

```python
python -m laws_sim.cli --real-yolo --image-dir /path/to/frames --steps 200
```

### Demo adversarial patch (standalone)

**Da immagine:**
```python
python -m laws_sim.cli --demo-patch foto.jpg
```

**Da webcam:**
```python
python -m laws_sim/cli.py --demo-patch webcam
```

Opzione: `--patch-steps 100` (default 80).

### Simulazione con patch pre‑ottimizzata

```python
python -m laws_sim/cli.py --real-yolo --patch care_kit_patch.pt --steps 200
```

## Opzioni CLI

| Opzione | Descrizione |
|---------|-------------|
| `--steps N` | Numero di step di simulazione (default 1000) |
| `--seed N` | Seme random (default 42) |
| `--verbose` | Log dettagliato per ogni step |
| `--no-plot` | Disabilita generazione grafici |
| `--real-yolo` | Usa YOLOv8 reale (richiede `yolov8n.pt`) |
| `--image-dir PATH` | Directory con frame reali per YOLO |
| `--demo-patch IMAGE` | Esegue demo patch su immagine o `webcam` |
| `--patch FILE.pt` | Carica patch pre‑ottimizzata per la simulazione |
| `--patch-steps N` | Step di ottimizzazione per la demo patch |

## File generati

- `laws_sim_v3_results.json` – metriche per ogni scenario (precision, recall, F1, CLAE).
- `laws_sim_v3_results.png` – grafici comparativi.
- `care_kit_patch.pt` – tensore della patch ottimizzata (demo).
- `patch_result.png` e `patch_loss.png` – visualizzazione della demo.

## Struttura del progetto

```
laws_sim/
├── __init__.py
├── config.py              # Costanti e riferimenti scientifici
├── entities.py            # Entità e ambiente
├── detection.py           # Visione (YOLO + modello analitico)
├── patch_optimizer.py     # Ottimizzazione patch con EoT
├── fusion_decision.py     # Fusione bayesiana e decisione
├── metrics.py             # Metriche e CLAE
├── simulator.py           # Loop principale
├── utils.py               # Console e plotting
└── cli.py                 # CLI entry point
```

## Riferimenti scientifici

- **YOLOv8 outdoor confidence 99.1%** – Sodhro et al., 2025. [DOI:10.1016/j.iot.2025.101707](https://doi.org/10.1016/j.iot.2025.101707)
- **Expectation over Transformation (EoT)** – Athalye et al., 2017. [arXiv:1707.07397](https://arxiv.org/abs/1707.07397)
- **Specifiche droni tattici** – Threod EOS C (61 km/h), RemoEye‑002B (75 km/h)
- **OSINT risk scoring** – GEN‑TPRM (Asili & Bahtiyar, 2025), ETIP (González‑Granadillo et al., 2021)

## Esempio di flusso completo

```bash
# Genera patch dalla webcam
python laws_sim/cli.py --demo-patch webcam --patch-steps 1000

# Simula con YOLO reale e patch
python laws_sim/cli.py --real-yolo --patch care_kit_patch.pt --steps 1000

# Leggi i risultati
cat laws_sim_v3_results.json
```
