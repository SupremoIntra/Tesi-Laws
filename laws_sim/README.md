# LAWS-SIM

Framework di ricerca per la valutazione della robustezza adversarial in
sistemi di percezione autonoma multi-dominio (Adversarial Robustness in
Computer Vision + PyTorch MPS). Tesi Magistrale in Sicurezza Informatica LM-66.

> Uso a scopo didattico e sperimentale. Tutte le entità (target, civili,
> profili OSINT) sono sintetiche; nessun dato reale, nessuna arma reale,
> nessun hardware di volo reale è coinvolto. Il framework esiste per
> **misurare la fragilità** di un ipotetico sistema di targeting
> automatizzato sotto attacco, non per costruirne uno operativo.

## Architettura del progetto (3 layer disaccoppiati)

1. **Vision** — Universal Adversarial Patch (EoT completo, 16 trasformazioni)
   addestrata contro YOLOv8n su VisDrone-DET, per misurare il degrado
   empirico della detection sulla classe "person". Metriche valutate:
   sensitività (R1), specificità (R2), media geometrica √(R1·R2), evasion
   rate, F1 — tutte con intervalli di confidenza bootstrap pre/post attacco.
2. **Simulation** — ambiente multi-agente che legge il drop empirico
   da `outputs/metrics/vision_metrics.json` e lo inietta nel loop tattico
   ("flickering" del sensore).
3. **Metadata Integrity** — Bayesian Sensor Fusion (Vision/OSINT/Behavioral)
   sotto attacco di Data Poisoning OSINT, valutato con la metrica proposta
   CEAE (Cost-Effective Adversarial Engagement).

## Struttura cartelle

```
laws_sim/
├── config.py               # Costanti derivate dai paper citati (vedi docstring)
├── cli.py                  # Entry point — orchestratore I/O
├── requirements.txt
├── src/                    # Moduli applicativi
│   ├── entities.py         # Environment, SimEntity, OSINTProfile (Bayes)
│   ├── detection.py        # Sensore visivo (Digital Twin, empirico F1-based)
│   ├── patch_optimizer.py  # Training Universal Adversarial Patch (EoT)
│   ├── visdrone_loader.py  # DataLoader VisDrone2019-DET
│   ├── fusion_decision.py  # Fusion Agent + Decision Agent (soglie IHL)
│   ├── simulator.py        # Loop multi-agente + evaluate_on_dataset
│   ├── metrics.py          # Metriche (R1/R2/gmean/F1/evasion) + bootstrap CI
│   └── utils.py            # Console (rich)
├── tools/                  # Script di analisi e valutazione (vedi sotto)
│   ├── annotate_mioDS.py         # Auto-annotatore per Domain Adaptation (Wu et al.)
│   ├── bootstrap_ci_report.py    # CI bootstrap pre/post + delta appaiato
│   ├── stratify_by_size.py       # Evasion rate stratificato per altezza bbox
│   ├── plot_k_selection.py       # Scelta empirica di K (F1/gmean/R1/R2 vs K)
│   └── plot_runs_comparison.py   # Confronto F1/evasion tra i run di training
├── data/visdrone/           # images/ + annotations/
└── outputs/
    ├── checkpoints/         # checkpoint_patch.pt (resumabile)
    ├── patches/             # care_kit_patch_universal.pt
    └── metrics/             # vision_metrics.json, training_metrics.json, full_report.json
```

## Installazione

```bash
pip install -r requirements.txt
```

Scarica VisDrone2019-DET-val (~370MB, bastano le immagini di test) da
https://github.com/VisDrone/VisDrone-Dataset e posizionalo in
`data/visdrone/images/` + `data/visdrone/annotations/`.

## Comandi CLI (`cli.py`)

### 1. Training della Universal Adversarial Patch

```bash
python cli.py --train-patch
```
Salva la patch in `outputs/patches/care_kit_patch_universal.pt` e i
checkpoint resumabili in `outputs/checkpoints/`. Interrompibile con `Ctrl+C`
(salvataggio di emergenza automatico).

### 2. Validazione empirica su VisDrone (YOLOv8 reale + patch)

```bash
python cli.py --eval-vision data/visdrone --patch outputs/patches/care_kit_patch_universal.pt
```
Calcola Precision/Recall/F1 frame-level con la patch iniettata
proporzionalmente (`get_chest_bbox_proportional`, Tactical Vest Assumption
50%×40%) e salva il risultato in `outputs/metrics/vision_metrics.json` —
il bridge letto dal simulatore. Opzionale: `--max-samples N`.

### 3. Report Vision consolidato pre/post (raccomandato per la tesi)

```bash
python cli.py --eval-report --patch outputs/patches/care_kit_patch_universal.pt
```
In un solo comando esegue due passate (PRE senza patch, POST con patch)
sullo stesso valset e produce la tabella pronta per la tesi/slide:
per ogni metrica (evasion rate, sensitività R1, specificità R2,
√(R1·R2), F1) mostra valore PRE e POST con CI 95%, il delta appaiato con
CI, il p-value bootstrap e il verdetto di significatività. Salva tutto in
`outputs/metrics/full_report.json`.

Opzioni: `--data DIR` (default `data/visdrone_val`), `--n-iter N` (default
10000), `--full-report` (aggiunge stratificazione per taglia e grafici K
rilanciando i tool dedicati).

### 4. Simulazione tattica multi-agente

```bash
python cli.py --run-sim
```
Esegue due esperimenti (Vision-only isolato, e Sistema Completo con Fusion
Bayesiana su 4 scenari: Baseline, Patch, OSINT Poisoning, Cascading) e
stampa le tabelle comparative (F1, FPR, CEAE) via `rich`. Se
`outputs/metrics/vision_metrics.json` non esiste, usa una baseline
fittizia di sicurezza (F1=0.710, Sodhro et al. 2025).

## Strumenti di analisi (`tools/`)

Ogni script è eseguibile in autonomia; `--eval-report --full-report` li
orchestra in sequenza.

```bash
# Intervalli di confidenza bootstrap pre/post (F1 e media geometrica),
# con verdetto a CI indipendenti + delta appaiato + p-value
python tools/bootstrap_ci_report.py --data data/visdrone_val \
    --patch outputs/patches/care_kit_patch_universal.pt --n-iter 10000

# Evasion rate stratificato per altezza del bounding box (diagnosi del
# soffitto strutturale: dipende dalla dimensione del bersaglio?)
python tools/stratify_by_size.py --data data/visdrone_val \
    --patch outputs/patches/care_kit_patch_universal.pt

# Scelta empirica di K: grafici F1/√(R1·R2)/R1/R2 vs K (ispezione visiva).
# Lanciare SENZA --max-samples per i numeri definitivi da tesi.
python tools/plot_k_selection.py --data data/visdrone_val

# Confronto F1/evasion rate tra i run di training (bar chart).
# Aggiornare la lista RUNS nello script ad ogni nuovo run.
python tools/plot_runs_comparison.py

# Domain Adaptation (Piano B): annota foto raw in formato VisDrone
python tools/annotate_mioDS.py
```

## File generati

- `outputs/checkpoints/checkpoint_patch.pt` — checkpoint training (resumabile)
- `outputs/patches/care_kit_patch_universal.pt` — patch finale
- `outputs/metrics/training_metrics.json` — storia loss/TV/confidence
- `outputs/metrics/vision_metrics.json` — F1/Precision/Recall empirici (bridge)
- `outputs/metrics/full_report.json` — report pre/post consolidato (tutte le metriche + CI + delta + p-value)
- `outputs/metrics/k_selection_plots.png` — grafici scelta di K
- `outputs/metrics/runs_comparison.png` — confronto tra run di training

## Riferimenti scientifici

Vedi docstring completo in `config.py` per la lista di tutte le fonti con
DOI/URL (Sodhro et al. 2025, Threod EOS-C, RemoEye-002B, GEN-TPRM,
ETIP, Carlini & Wagner 2017, Thys et al. 2019, Brown et al. 2017,
Athalye et al. 2017, Arkin 2009).

## Note per la commissione

Il capitolo "Limitations and Future Work" menziona: (1) il Physical Domain Gap (tutto l'attacco è digitale,
non testato su stampa CMYK reale); (2) la stabilità di `F.grid_sample`
su backend MPS in sessioni di training molto lunghe; (3) la necessità di
calibrazione fine dell'`ENGAGEMENT_THRESHOLD` nel Fusion Agent multi-agente;
(4) il campione ridotto di negativi veri (9 frame su VisDrone-val) che
rende la stima della specificità R2 poco potente rispetto alla
sensitività R1 (80 frame).
