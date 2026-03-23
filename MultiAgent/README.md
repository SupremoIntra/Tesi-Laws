# LAWS-SIM v1.0

**Multi-Agent LAWS Vulnerability Simulator**  
Simulatore per tesi sperimentale su Lethal Autonomous Weapon Systems (LAWS).

---

## Installazione rapida

```bash
pip install -r requirements.txt
python laws_sim.py
```

## Uso

```bash
# Simulazione base (100 step, seed 42)
python laws_sim.py

# 200 step, seed diverso, log verboso sui target
python laws_sim.py --steps 200 --seed 7 --verbose

# Senza plot (server/CI)
python laws_sim.py --no-plot
```

## Architettura agenti

```
[OSINTAgent]   → profilo sintetico (social, geo, network)
    ↓
[VisionAgent]  → YOLO-sim con EoT Adversarial Patch
    ↓
[FusionAgent]  → fusione Bayesiana pesata (3 layer)
    ↓
[DecisionAgent]→ soglie engagement + IHL check
```

## Scenari di attacco

| Scenario | Attacco | Layer coinvolto |
|---|---|---|
| Baseline | Nessuno | — |
| Adversarial Patch | EoT patch su CARE kit | Vision |
| OSINT Poisoning | Data poisoning profilo | OSINT |
| Cascading | Patch + poisoning | Vision + OSINT |

## Output

- **Terminale**: tabella metriche (Precision/Recall/F1/FPR/CLAE)
- `laws_sim_results.json`: export numerico completo
- `laws_sim_results.png`: plot comparativo (se matplotlib disponibile)

## Metrica CLAE

**Cross-Layer Attack Efficiency** (proposta originale):

```
CLAE = ΔF1 / C_attack
```

Misura il degradamento normalizzato per il costo dell'attacco.
Un CLAE alto indica un attacco efficiente a basso costo.

## Collegamento CARE kit

Il flag `care_kit_active = True` su un'entità attiva il modello EoT:

```python
conf_patch = conf_base × (1 − S × exp(−λ × d) + ε)
```

- `S = 0.65` — soppressione base della patch
- `λ = 0.04` — decay con distanza (patch visualmente piccola da lontano)
- `ε ~ N(0, 0.08)` — rumore fisico (illuminazione, angolo, blur)
