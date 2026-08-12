<div align="center">

# LAWS-SIM

**Valutazione della robustezza avversariale in sistemi di percezione autonoma multi-dominio**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/detector-YOLOv8n-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Datasets](https://img.shields.io/badge/datasets-VisDrone%20%7C%20Okutama-lightgrey.svg)](#dataset)
[![Status](https://img.shields.io/badge/status-tesi%20magistrale-orange.svg)](#)

Tesi Magistrale in Sicurezza Informatica (LM-66)

</div>

---

> [!IMPORTANT]
> **Uso didattico e sperimentale.** Tutte le entità del simulatore (bersagli,
> civili, profili OSINT) sono sintetiche: nessun dato personale reale, nessuna
> arma reale, nessun hardware di volo reale è coinvolto. Il framework esiste per
> **misurare la fragilità** di un ipotetico sistema di targeting automatizzato
> sotto attacco, non per costruirne uno operativo.

## Indice

- [Cosa fa](#cosa-fa)
- [Architettura](#architettura)
- [Installazione](#installazione)
- [Dataset](#dataset)
- [Uso](#uso)
- [Strumenti di analisi](#strumenti-di-analisi)
- [Artefatti generati](#artefatti-generati)
- [Limitazioni note](#limitazioni-note)
- [Riferimenti](#riferimenti)
- [Licenza](#licenza)

## Cosa fa

LAWS-SIM ottimizza una **Universal Adversarial Patch** contro un rilevatore di
persone reale (YOLOv8n) su immagini aeree, misura il degrado di prestazione con
intervalli di confidenza bootstrap, e propaga quel degrado misurato in un
simulatore multi-agente per osservarne l'effetto su una decisione automatizzata.

Il modello di minaccia è un **physical evasion attack in fase di inferenza**, in
regime *transfer*: l'avversario non ha accesso all'infrastruttura logica né ai
pesi del rilevatore in esercizio, e il suo unico vettore è una superficie
stampata applicata al corpo.

## Architettura

Tre livelli deliberatamente disaccoppiati, con statuti epistemici diversi.

| Livello | Contenuto | Statuto dei risultati |
|---|---|---|
| **1 — Vision** | Patch universale con EoT (16 trasformazioni) contro YOLOv8n su VisDrone-DET e Okutama-Action. Metriche: sensitività `R1`, specificità `R2`, media geometrica `√(R1·R2)`, evasion rate, `F1`, tutte con CI bootstrap pre/post | **Misura sperimentale** |
| **2 — Simulation** | Ambiente multi-agente. Il canale visivo è un modello di Bernoulli parametrizzato su `R1`/`R2` letti da `vision_metrics.json`; la rete neurale non viene mai caricata | Conseguenza di un modello |
| **3 — Metadata Integrity** | Fusione bayesiana (Vision 0.45 / OSINT 0.35 / Behavioral 0.20) sotto *data poisoning* OSINT, con vincoli precauzionali a valle | Conseguenza di un modello |

Il disaccoppiamento passa per un unico artefatto su disco,
`outputs/metrics/vision_metrics.json`: quattro parametri misurati fuori linea.
Ciò rende i due livelli verificabili in modo indipendente e rende esplicito
l'insieme delle informazioni che transitano dall'uno all'altro.

```
laws_sim/
├── config.py                     # Costanti e iperparametri, con fonti nei docstring
├── cli.py                        # Entry point, orchestratore I/O
├── requirements.txt
├── src/
│   ├── entities.py               # Environment, SimEntity, OSINTProfile (Bayes)
│   ├── detection.py              # Canale visivo statistico (Bernoulli su R1/R2)
│   ├── patch_optimizer.py        # Ottimizzazione della patch universale (EoT, top-K, TV)
│   ├── visdrone_loader.py        # Loader VisDrone2019-DET
│   ├── okutama_loader.py         # Loader Okutama-Action (label native 3840x2160)
│   ├── fusion_decision.py        # FusionAgent + DecisionAgent (soglie e vincoli)
│   ├── simulator.py              # Loop multi-agente + evaluate_on_dataset
│   ├── metrics.py                # R1/R2/gmean/F1/evasion + bootstrap CI + CEAE
│   └── utils.py                  # Console (rich)
├── tools/
│   ├── bootstrap_ci_report.py    # CI bootstrap pre/post + delta appaiato + p-value
│   ├── stratify_by_size.py       # Evasion rate stratificato per altezza bbox
│   ├── count_negative_candidates.py  # Conteggio dei negativi veri nel valset
│   ├── plot_k_selection.py       # Curve F1 / √(R1·R2) / R1 / R2 al variare di K
│   ├── plot_CI_box.py            # Forest plot degli intervalli di confidenza
│   ├── plot_training_curves.py   # Curve di loss / TV / grad-norm / learning rate
│   ├── plot_runs_comparison.py   # Confronto tra configurazioni di training
│   ├── generate_before_after_images.py  # Coppie PRE/POST con box disegnate
│   ├── debug_fusion_trace.py     # Traccia passo-passo della fusione bayesiana
│   └── annotate_mioDS.py         # Auto-annotatore per Domain Adaptation (non usato)
├── data/
│   ├── visdrone_val/             # images/ + annotations/
│   └── okutama_train/            # Frames/ + Labels/SingleActionLabels/3840x2160/
└── outputs/
    ├── checkpoints/              # checkpoint_patch.pt (resumabile)
    ├── patches/                  # care_kit_patch_universal.pt
    └── metrics/                  # vedi "Artefatti generati"
```

## Installazione

```bash
git clone https://github.com/intradaniele/Tesi-Laws.git
cd Tesi-Laws/laws_sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Sviluppato e testato su **Apple Silicon M4, 16 GB di memoria unificata**, backend
PyTorch MPS. Il campionamento su griglia è mantenuto in `float32`: prove in
precisione ridotta hanno evidenziato instabilità numerica nel gradiente di
`F.grid_sample` su questo backend. L'operatore all'indietro non ha
implementazione nativa MPS ed è eseguito su CPU tramite
`PYTORCH_ENABLE_MPS_FALLBACK=1`, impostato automaticamente da `cli.py`.

## Dataset

Nessuno dei due dataset è incluso nel repository. Vanno scaricati dalle fonti
originali e ne valgono le rispettive licenze d'uso, entrambe **limitate alla
ricerca**.

| Dataset | Uso | Fonte |
|---|---|---|
| VisDrone2019-DET (val) | Sviluppo, prima campagna | [VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) |
| Okutama-Action | Campagna finale | [okutama-action.org](http://okutama-action.org/) |

> [!NOTE]
> Le label di Okutama sono in spazio nativo **3840×2160**, non 1280×720. Lo
> scaling verso il canvas di rete avviene in un solo passaggio in
> `okutama_loader.get_sample`.

## Uso

### Ottimizzazione della patch

```bash
python cli.py --train-patch
```

Interrompibile con `Ctrl+C` (salvataggio d'emergenza) e ripartibile dal
checkpoint. Salva in `outputs/patches/care_kit_patch_universal.pt`.

> [!WARNING]
> Il file salvato al termine è la patch dell'**ultimo aggiornamento**, non quella
> con la media mobile di loss migliore: `cli.py` sovrascrive con
> `results["patch"]` lo stesso percorso usato durante il training per il
> checkpoint migliore. Nei run condotti la differenza è trascurabile (media
> mobile 0.7574 al minimo contro 0.7583 finale), ma il nome del file è
> fuorviante.

### Validazione empirica

```bash
python cli.py --eval-vision data/okutama_train \
    --patch outputs/patches/care_kit_patch_universal.pt --img-size 960
```

Calcola le metriche a livello di fotogramma con la patch composta
proporzionalmente sul bersaglio (`get_chest_bbox_proportional`, 50% × 40% del
bounding box, 20% dell'area) e scrive `outputs/metrics/vision_metrics.json`, il
ponte letto dal simulatore. Opzionale `--max-samples N`.

### Report consolidato pre/post — raccomandato

```bash
python cli.py --eval-report \
    --patch outputs/patches/care_kit_patch_universal.pt --n-iter 10000
```

Due passate sullo stesso insieme di validazione (PRE senza patch, POST con
patch) e, per ogni metrica, valore PRE e POST con CI 95%, delta appaiato con CI,
p-value bootstrap e verdetto di significatività. Con `--full-report` orchestra
anche stratificazione per taglia e grafici di selezione di K.

### Simulazione multi-agente

```bash
python cli.py --run-sim
```

Due esperimenti — canale visivo isolato, e sistema completo con fusione
bayesiana su quattro scenari (Baseline, Patch, OSINT Poisoning, Cascading) — con
tabelle comparative di F1, FPR e CEAE. In assenza di
`outputs/metrics/vision_metrics.json` usa una baseline di sicurezza
`R1 = 0.710` (Sodhro et al. 2025) e lo segnala a schermo.

## Strumenti di analisi

Ogni script è eseguibile in autonomia; `--eval-report --full-report` li orchestra
in sequenza.

```bash
# CI bootstrap pre/post con delta appaiato e p-value
python tools/bootstrap_ci_report.py --data data/okutama_train \
    --patch outputs/patches/care_kit_patch_universal.pt --n-iter 10000

# Evasion rate stratificato per altezza del bounding box
python tools/stratify_by_size.py --data data/okutama_train \
    --patch outputs/patches/care_kit_patch_universal.pt

# Selezione empirica di K: curve F1 / gmean / R1 / R2 al variare di K.
# Lanciare SENZA --max-samples per i valori definitivi.
python tools/plot_k_selection.py --data data/okutama_train

# Curve di training (loss, TV, norma del gradiente, learning rate)
python tools/plot_training_curves.py
```

## Artefatti generati

Dal momento in cui è stata adottata la convenzione di naming, gli artefatti di
analisi seguono lo schema `<base>_<loader>[_<img_size>][_stride<N>].ext`, per
evitare che run diversi si sovrascrivano silenziosamente.

| Percorso | Contenuto |
|---|---|
| `outputs/checkpoints/checkpoint_patch.pt` | Checkpoint di training, ripartibile |
| `outputs/patches/care_kit_patch_universal.pt` | Patch al termine del budget di ottimizzazione |
| `outputs/metrics/vision_metrics.json` | **Ponte di pipeline**: `R1`/`R2` pre e post, letti dal simulatore. Nome fisso per scelta: esiste un solo risultato attivo corrente |
| `outputs/metrics/training_metrics.json` | Storia di loss, TV, learning rate, norma del gradiente |
| `outputs/metrics/full_report_<loader>_<size>_stride<N>.json` | Report pre/post con CI, delta e p-value |
| `outputs/metrics/k_selection_plots[_<loader>].png` | Grafici di selezione di K |
| `outputs/metrics/archive_fase6_visdrone/` | Artefatti storici con nomi generici, precedenti alla convenzione |

## Limitazioni note

1. **Physical domain gap** — l'attacco è interamente digitale. Non è stato
   testato su stampa reale, e la penalizzazione di variazione totale è una
   condizione necessaria ma non sufficiente alla riproducibilità fisica.
2. **Regime transfer non quantificato** — la patch è ottimizzata in *white-box*
   su un modello surrogato. Le misure sono un limite superiore all'efficacia,
   non una stima operativa.
3. **Campione di negativi veri ridotto** — 9 fotogrammi su VisDrone-val contro
   80 positivi: la stima di `R2` è molto meno potente di quella di `R1`.
4. **Interazione fra risoluzione del canvas e stride del rilevatore** — la
   maschera spaziale marca una cella solo se il suo centro cade nel bounding
   box, e gli stride sono 8/16/32 px. Su canvas fortemente ridotti i bersagli
   più sottili non generano alcuna cella e i relativi passi vengono scartati
   prima dell'aggiornamento. Effetto misurato, non ancora attribuito
   quantitativamente.
5. **Programma del passo di apprendimento** — `T_max` è calcolato sul numero di
   aggiornamenti *attesi* (`n_steps // accumulo`). Poiché i passi senza
   bersagli utilizzabili vengono scartati, il numero realizzato è inferiore e
   l'annealing non raggiunge `eta_min` entro la durata del run.
6. **Stabilità di `F.grid_sample` su MPS** in sessioni di training molto lunghe.

## Riferimenti

Le fonti complete con DOI sono nei docstring di `config.py`. Le principali:

- Carlini & Wagner (2017), *Towards Evaluating the Robustness of Neural
  Networks* — riparametrizzazione per cambio di variabile
- Brown et al. (2017), *Adversarial Patch* — paradigma della perturbazione localizzata
- Athalye et al. (2018, ICML), *Synthesizing Robust Adversarial Examples* — EoT
- Thys et al. (2019), *Fooling automated surveillance cameras* — TV loss, aggregazione top-K
- Zhu et al. (2021), *Detection and Tracking Meet Drones Challenge* — VisDrone
- Barekatain et al. (2017), *Okutama-Action* — dataset aereo
- Sodhro et al. (2025), *Internet of Things* 33:101707 — baseline YOLOv8
- Shrestha et al. (2023, IROS) — trasferibilità di patch avversariali nel dominio UAV

## Licenza

Distribuito sotto **GNU Affero General Public License v3.0** — vedi
[`LICENSE`](LICENSE).

La scelta non è discrezionale: questo progetto usa Ultralytics YOLOv8, rilasciato
sotto AGPL-3.0, licenza che estende l'obbligo di rilascio del codice sorgente
all'intera opera derivata. Le licenze permissive (MIT, BSD) e le licenze
Creative Commons non sarebbero conformi.

Le licenze dei dataset sono indipendenti da questa e restano quelle dei
rispettivi detentori: nessun frame, annotazione o peso derivato da VisDrone o
Okutama-Action è incluso nel repository.
