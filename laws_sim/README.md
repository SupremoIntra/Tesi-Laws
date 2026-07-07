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
   addestrata contro YOLOv8n su VisDrone-DET, per misurare il drop empirico
   di F1-Score sulla classe "person".
2. **Simulation** — ambiente multi-agente che legge il drop di F1 empirico
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
│   └── utils.py            # Console (rich)
├── tools/
│   └── annotate_mioDS.py   # Auto-annotatore per Domain Adaptation (Wu et al.)
├── data/visdrone/           # images/ + annotations/ 
└── outputs/
    ├── checkpoints/         # checkpoint_patch.pt (resumabile)
    ├── patches/             # care_kit_patch_universal.pt
    └── metrics/             # vision_metrics.json, training_metrics.json
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
il bridge letto dal simulatore.

Opzionale: `--max-samples N` per limitare i frame testati.

### 3. Simulazione tattica multi-agente

```bash
python cli.py --run-sim
```
Esegue due esperimenti (Vision-only isolato, e Sistema Completo con Fusion
Bayesiana su 4 scenari: Baseline, Patch, OSINT Poisoning, Cascading) e
stampa le tabelle comparative (F1, FPR, CEAE) via `rich`.

Se `outputs/metrics/vision_metrics.json` non esiste, usa una baseline
fittizia di sicurezza (F1=0.710, Sodhro et al. 2025).

### 4. Domain Adaptation (Piano B — dataset custom)

```bash
python tools/annotate_mioDS.py
```
Genera annotazioni in formato VisDrone da foto raw scattate dall'utente
(cartella `custom_dataset/images/`), per validare l'ipotesi di
*Targeted Threat Modeling* (Wu et al., 2020) su un dominio image-specific.

## File generati

- `outputs/checkpoints/checkpoint_patch.pt` — checkpoint training (resumabile)
- `outputs/patches/care_kit_patch_universal.pt` — patch finale
- `outputs/metrics/training_metrics.json` — storia loss/TV/confidence
- `outputs/metrics/vision_metrics.json` — F1/Precision/Recall empirici (bridge)

## Riferimenti scientifici

Vedi docstring completo in `config.py` per la lista di tutte le fonti con
DOI/URL (Sodhro et al. 2025, Threod EOS-C, RemoEye-002B, GEN-TPRM,
ETIP, Carlini & Wagner 2017, Thys et al. 2019, Brown et al. 2017,
Athalye et al. 2017, Arkin 2009).

## Note per la commissione

Il capitolo "Limitations and Future Work" della tesi deve menzionare
esplicitamente: (1) il Physical Domain Gap (tutto l'attacco è digitale,
non testato su stampa CMYK reale); (2) la stabilità di `F.grid_sample`
su backend MPS in sessioni di training molto lunghe; (3) la necessità di
calibrazione fine dell'`ENGAGEMENT_THRESHOLD` nel Fusion Agent multi-agente.
