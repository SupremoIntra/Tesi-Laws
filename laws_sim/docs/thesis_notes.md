# Note di progetto — LAWS-SIM

Log delle decisioni tecniche e architetturali. Aggiornato a ogni modifica
importante. Scopo: avere una cronologia leggibile da riusare nella tesi,
non un log di sessioni.

---

## Architettura

Tre layer disaccoppiati, collegati da file JSON (non da chiamate dirette):

1. **Vision** (`patch_optimizer.py`) — training di una Universal Adversarial
   Patch contro YOLOv8n, EoT completo (16 trasformazioni), loss asintotica.
2. **Simulation** (`simulator.py`, `entities.py`, `detection.py`) — ambiente
   multi-agente che legge il drop di F1 empirico da `vision_metrics.json`.
3. **Fusion/Decision** (`fusion_decision.py`) — sensor fusion bayesiano
   (Vision/OSINT/Behavioral) con soglie di ingaggio IHL.

Il ponte JSON è deliberato: i bug di visione non bloccano il simulatore
tattico, e permette esperimenti isolati (es. solo OSINT).

## Decisioni chiave e perché

- **Loss asintotica** (`-log(1 - mean(conf) + eps)`) invece di Hinge Loss:
  la Hinge con soglia fissa (`relu(score - 0.10)`) ha derivata zero sotto
  soglia → gradiente morto. L'asintotica non ha soglia, converge sempre.
- **TV_WEIGHT = 0.1** (non 0.0001): un TV loss basso produce rumore ad alta
  frequenza che l'interpolazione bilineare di `grid_sample` nell'EoT
  distrugge. Un peso alto forza pattern a bassa frequenza che sopravvivono
  (Thys et al. 2019).
- **Parametrizzazione sigmoide** dei pixel patch (non `clamp`): clamp
  uccide il gradiente ai pixel che toccano 0 o 1.
- **PATCH_BBOX_COVERAGE = 0.20**: coerente con `get_chest_bbox_proportional`
  (50% larghezza × 40% altezza = 0.20 dell'area del bersaglio).
- **N_TARGETS=3, N_CIVILIANS=30**: rapporto 1:10, scenario "Urban Clutter"
  (Arkin 2009) per stressare il Fusion Agent sotto OSINT poisoning verso
  falsi positivi sui civili.
- **ENGAGEMENT_THRESHOLD=0.58 / ALERT=0.38 / TRACK=0.22**: la soglia di
  ingaggio volutamente vicina al 50% modella l'abbassamento degli standard
  di certezza operativa osservato in scenari reali (Automation Bias).
- **Train/val split VisDrone separati**: trainset (6471 img) per
  `--train-patch`, valset (531 img, quello del baseline storico F1=0.760)
  riservato a `--eval-vision`. Allenare e valutare sulle stesse immagini
  avrebbe inflazionato l'evasion rate misurato.
- **Nessun modello analitico di decadimento distanza-based** (scartato
  `PATCH_SUPPRESSION`/`PATCH_DIST_FALLOFF`): il degrado visivo è solo
  data-driven, dal drop di F1 empirico.

## Bug risolti

| # | Problema | Causa | Fix |
|---|---|---|---|
| 1 | `grid_sampler_2d_backward` non implementato su MPS | EoT usa `F.grid_sample` per rotazione/scala | `PYTORCH_ENABLE_MPS_FALLBACK=1`: solo quell'operatore va su CPU, il resto resta su MPS |
| 2 | `RuntimeError: view size is not compatible...` nel primo conv2d di YOLO | `.expand()` + `.permute()` lasciano un tensore non contiguo, MPS non lo tollera (CPU/CUDA sì) | `.contiguous()` dopo il permute dell'immagine e sul batch finale |
| 3 | `config.py` senza `N_TARGETS`, `FUSION_WEIGHTS`, soglie ecc. | Costanti perse in un refactor precedente | Ricostruite con fonte citata per ognuna |
| 4 | Nessuna detection nel simulatore (TP/FP sempre 0) | `DRONE_ALTITUDE_M=300` sempre oltre `YOLO_MAX_RANGE=150` | Altitudine realistica: 10m |
| 5 | Patch copriva l'intera figura (incluso volto) | `PATCH_H/W` fissi in pixel assoluti | `get_chest_bbox_proportional`: patch scalata sul bbox reale della persona |
| 6 | Vanishing gradient con Hinge Loss + soglia 0.10 | derivata zero sotto soglia | Sostituita con loss asintotica (vedi sopra) |

## Letteratura citata (riferimento rapido)

- Sodhro et al. 2025 — baseline YOLOv8 outdoor 99.1% confidence
- Carlini & Wagner 2017 — margine di robustezza nella loss
- Wu et al. 2020 — Domain-Specific Attacks > Universal Attacks
- Athalye et al. 2017 — EoT
- Thys et al. 2019 — TV Loss per pattern stampabili
- Brown et al. 2017 — Adversarial Patch, convergenza rapida
- Arkin 2009 — Governing Lethal Behavior, Principio di Distinzione IHL

DOI/URL completi in `config.py`.

## Stato attuale

Pipeline validata end-to-end (smoke test su Mac M4, MPS, dataset sintetico):
nessun crash, gradiente scorre, checkpoint funzionante. Pronta per il
training reale su VisDrone train split.

## Aperto / da fare

- Verificare stabilità memoria MPS su run lungo (1500 step reali)
- Custom dataset (Domain Adaptation, `tools/annotate_mioDS.py`) — da
  eseguire dopo il training universale, se serve un piano B
- Calibrazione fine di `ENGAGEMENT_THRESHOLD` dopo aver visto i risultati
  del Fusion Agent con F1 reale
- Physical Domain Gap (stampa CMYK reale, drone reale) — limitazione da
  menzionare in tesi, non da risolvere nel codice
