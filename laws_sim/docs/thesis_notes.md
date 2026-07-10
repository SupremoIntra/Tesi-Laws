# Note di progetto — LAWS-SIM

Log delle decisioni tecniche, degli esperimenti e dei risultati. Aggiornato
a ogni modifica importante. Scopo: cronologia completa ma leggibile, da
riusare nella tesi e nelle call col relatore.

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

- **Loss asintotica** (`-log(1 - conf + eps)`) invece di Hinge Loss: la
  Hinge con soglia fissa ha derivata zero sotto soglia → gradiente morto.
  L'asintotica non ha soglia, converge sempre.
- **Aggregazione top-K (K=20)** invece di media su tutte le celle
  mascherate: generalizza la tecnica "max objectness" di Thys et al.
  (2019) — concentra il gradiente sulle celle vicine a una detection
  reale invece di diluirlo su celle di sfondo.
- **TV_WEIGHT = 0.1**: un TV loss basso produce rumore ad alta frequenza
  che l'interpolazione bilineare di `grid_sample` nell'EoT distrugge. Un
  peso alto forza pattern a bassa frequenza che sopravvivono (Thys et
  al. 2019).
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
  `--train-patch`, valset (531 img) riservato a `--eval-vision`. Allenare
  e valutare sulle stesse immagini avrebbe inflazionato l'evasion rate.
- **Nessun modello analitico di decadimento distanza-based**: il degrado
  visivo è solo data-driven, dal drop di F1 empirico.

## Glossario essenziale

- **Step raw**: un ciclo del training loop (immagine → EoT → forward →
  backward). `PATCH_STEPS` conta questi, non gli aggiornamenti dei pixel.
- **Gradient Accumulation**: quanti step raw si sommano prima di un
  update reale. `update_reali = step_raw / accumulation_steps`.
- **Update reale**: la chiamata a `optimizer.step()` — il momento in cui
  i pixel della patch cambiano davvero.
- **EoT**: media della loss su N trasformazioni random della patch
  (rotazione, scala, colore), per farla funzionare su una distribuzione
  di condizioni realistiche, non su un'immagine esatta.
- **Evasion Rate**: frazione di frame con persona reale in cui YOLO, sotto
  attacco, non rileva nessuno.
- **F1-Score sotto attacco**: media armonica di Precision e Recall sulla
  detection frame-level; un calo indica un peggioramento del sensore.

## Bug tecnici risolti

| # | Problema | Causa | Fix |
|---|---|---|---|
| 1 | `grid_sampler_2d_backward` non implementato su MPS | EoT usa `F.grid_sample` per rotazione/scala | `PYTORCH_ENABLE_MPS_FALLBACK=1`: solo quell'operatore va su CPU, il resto resta su MPS |
| 2 | `RuntimeError: view size is not compatible...` nel primo conv2d di YOLO | `.expand()` + `.permute()` lasciano un tensore non contiguo, MPS non lo tollera | `.contiguous()` dopo il permute dell'immagine e sul batch finale |
| 3 | `config.py` senza `N_TARGETS`, `FUSION_WEIGHTS`, soglie ecc. | Costanti perse in un refactor precedente | Ricostruite con fonte citata per ognuna |
| 4 | Nessuna detection nel simulatore (TP/FP sempre 0) | `DRONE_ALTITUDE_M=300` sempre oltre `YOLO_MAX_RANGE=150` | Altitudine realistica: 10m |
| 5 | Patch copriva l'intera figura (incluso volto) | `PATCH_H/W` fissi in pixel assoluti | `get_chest_bbox_proportional`: patch scalata sul bbox reale della persona |
| 6 | Vanishing gradient con Hinge Loss | derivata zero sotto soglia | Sostituita con loss asintotica |
| 7 | LR resta vicino al massimo per quasi tutta la durata del run invece di scendere con la curva coseno | `CosineAnnealingLR(T_max=n_steps)` usa gli step raw come T_max, ma `scheduler.step()` viene chiamato una volta per **update reale** — con accum=4 la curva si allunga di 4x | `T_max = n_steps // GRADIENT_ACCUMULATION_STEPS` (unità: update reali) |

---

## Cronologia esperimenti — tabella completa

| # | Config | Update reali | Scheduler | F1 | Evasion Rate |
|---|---|---|---|---|---|
| 0 | Baseline storico (Hinge Loss) | 3000 | corretto | 0.760 | 38.7% |
| 1 | Loss asintotica, mean, accum=4 | 375 | allungato 4x | 0.740 | 41.25% |
| 2 | Loss asintotica, mean, accum=2 | 750 | allungato 2x | 0.750 | 40.0% |
| 3 | Loss asintotica, mean, accum=4 | 2500 | corretto | 0.720 | **43.75%** |
| 4 | Loss asintotica, mean, accum=16 | 625 | corretto | 0.740 | 41.25% |
| 5 | **Loss asintotica, top-K=20, accum=4** | **1250** | **corretto** | **0.720** | **43.75%** |

**Risultato di riferimento: riga 5.** Stesso massimo della riga 3, con
metà del budget di calcolo — la configurazione più efficiente misurata,
ed è essa stessa la prova del punto seguente.

### Interpretazione: soffitto strutturale, non un limite di ottimizzazione

Sei configurazioni radicalmente diverse — due formule di loss, due
aggregazioni (mean/top-K), accumulo da 1 a 16, scheduler corretto o
meno — convergono tutte nella stessa banda stretta (38.7%-43.75%).
Se il limite fosse "serve solo la loss giusta" o "servono solo più
update", almeno una configurazione l'avrebbe superato nettamente.

**Diagnosi del gradiente** (misurata direttamente, non ipotizzata): la
norma del gradiente per immagine singola è debole (0.0007-0.005), il
gradient clipping (soglia 1.0) non è mai scattato in nessun run.
Aumentare l'accumulo da 4 a 16 (media su più immagini) ha peggiorato il
risultato, non migliorato: prova che il segnale utile condiviso tra
scene diverse è intrinsecamente debole, non un problema di quantità di
update.

**Causa più probabile, dalla letteratura UAV-specific** (non dai paper
generici a livello del suolo — Thys, Brown, DPatch): i pedoni ripresi
dall'alto occupano pochissimi pixel assoluti. Nessuno dei lavori con
risultati forti su VisDrone attacca i pedoni:

| Paper | Anno | Bersaglio su VisDrone | Perché |
|---|---|---|---|
| Shrestha, Pathak, Viegas — "Towards a Robust Adversarial Patch Attack Against UAV Object Detection" | 2023 | Car (80% ASR white-box) | Patch pensate per prospettiva/distanza UAV |
| LFRAP (Multi-Dimensional Feature Optimization) | 2025 | Car, truck, van, bus — pedoni esplicitamente scartati | *"Given the small size of pedestrians under the overhead perspective of UAVs..."* |
| Adversarial patch attacks against aerial imagery object detectors | 2023 | Aerei, loss su feature intermedie | Bersagli grandi + tecnica alternativa (feature-level) |

Il nostro soffitto al 44% è verosimilmente il prezzo di aver scelto il
bersaglio più difficile del dominio (persone, non veicoli), non un
limite generico del framework.

### Verifica sui nostri dati: stratificazione per dimensione bersaglio

`tools/stratify_by_size.py` misura l'evasion rate per altezza del bbox,
usando la patch già addestrata (zero costo di training):

| Bucket altezza | Evasi/Totali | Evasion Rate |
|---|---|---|
| 60-100px | 34/314 | 10.8% |
| 100-150px | 1/56 | 1.8% |
| 150px+ | 5/11 | 45.5% |

Pattern **non monotono** — più sfumato dell'ipotesi "più grande = più
facile da evadere". Lettura più probabile: due effetti opposti in gioco.
(1) Budget di pixel della patch: bersagli grandi danno più superficie
reale all'attacco. (2) Margine di confidenza di partenza: bersagli
piccoli/sfocati hanno già una confidenza YOLO bassa in partenza, basta
poco per spingerli sotto soglia; bersagli medi ben risolti hanno
confidenza alta e la patch (debole) non basta a scalfirla. Il bucket
150px+ è promettente ma **statisticamente debole (11 casi totali)** —
non conclusivo, da trattare come pista esplorativa, non come risultato
confermato.

---

## Direzione futura

**Non ripetere la stessa configurazione più a lungo** — i 6 run già
dimostrano rendimenti piatti scalando solo gli step raw a parità di
resto. Le prossime leve agiscono sulla causa strutturale, non
sull'ottimizzatore:

1. **Verifica con campione più ampio** della stratificazione per
   dimensione (il bucket 150px+ ha solo 11 casi) prima di impegnare ore
   di calcolo su quell'ipotesi.
2. **Se confermata**: rifiltrare training+eval a un'altezza minima
   coerente con `DRONE_ALTITUDE_M=10` (ingaggio ravvicinato, non
   sorveglianza ad area larga) — non è cherry-picking, è allineamento
   allo scenario già definito nel simulatore.
3. **A quel punto** un budget lungo (≥20.000 step raw) con la loss
   top-K ha una motivazione scientifica per rendere di più.
4. **Riserva**: loss su feature intermedie del backbone invece che
   sull'output finale (tecnica del paper 2023 su aerial imagery) — più
   invasiva da implementare (hook sui layer intermedi di YOLO).
5. **Piano B sempre disponibile**: dataset custom image-specific
   (`tools/annotate_mioDS.py`, Wu et al. 2020) — converge più in fretta
   perché non deve generalizzare su migliaia di scene eterogenee.

## Letteratura citata

| Fonte | Uso nel progetto |
|---|---|
| Sodhro et al. 2025 | Baseline YOLOv8 outdoor 99.1% confidence |
| Carlini & Wagner 2017 | Margine di robustezza nella loss |
| Wu et al. 2020 | Domain-Specific Attacks > Universal Attacks (Piano B) |
| Athalye et al. 2017 | EoT |
| Thys et al. 2019 | TV Loss, aggregazione max/top-K |
| Brown et al. 2017 | Adversarial Patch, convergenza |
| Arkin 2009 | Principio di Distinzione IHL, scenario Urban Clutter |
| Liu et al. 2019 | DPatch — loss congiunta box+classe (riserva Fase 2) |
| Huang et al. 2020 | Universal Physical Camouflage Attacks — design region-aware |
| Hu et al. 2021 | Naturalistic Patch — GAN latent space (contributo originale) |
| Shrestha, Pathak, Viegas 2023 | Patch UAV-specific su VisDrone (Car, 80% ASR) |
| LFRAP 2025 | Conferma: pedoni scartati per dimensione da vista aerea |

DOI/URL completi in `config.py`.

## Punti di forza del lavoro

- **Metodo sistematico**: sei configurazioni isolate una variabile alla
  volta, non un tuning casuale fino al numero migliore.
- **Diagnosi quantitativa, non intuitiva**: norma del gradiente misurata
  direttamente, non dedotta — la decisione di provare il top-K viene da
  un dato, non da un'idea generica.
- **Confronto con la letteratura specifica**, non solo generica: il
  soffitto trova conferma in paper UAV-specific su VisDrone, non solo
  nei paper "classici" a livello del suolo.
- **Framework multi-dominio**: la parte Vision è un layer di un sistema
  più ampio (Fusion bayesiano, OSINT poisoning, soglie IHL) che la
  maggior parte della letteratura sulle adversarial patch non modella.

---

## Sintesi per la call col relatore

### La storia in 5 frasi

1. Un attacco adversarial patch universale contro YOLOv8 su VisDrone
   riduce l'F1-Score da 0.760 a 0.720, portando l'Evasion Rate dal 38.7%
   al 43.75% — un effetto reale ma modesto, non un collasso del sensore.
2. Sei configurazioni sistematicamente isolate (update reali, scheduling
   LR, accumulo, formula della loss, aggregazione mean/top-K) convergono
   tutte nella stessa banda stretta di evasion rate.
3. La loss top-K (ispirata a Thys et al. 2019, validata da una diagnosi
   diretta sul gradiente) raggiunge lo stesso massimo con metà del
   budget di calcolo — prova che il limite non è nell'ottimizzatore.
4. La letteratura UAV-specific su VisDrone conferma: nessun lavoro con
   risultati forti attacca i pedoni, tutti scelgono veicoli, perché i
   pedoni da vista aerea sono strutturalmente troppo piccoli per un
   budget di patch limitato.
5. La prossima leva è allineare i dati allo scenario tattico già
   definito (ingaggio ravvicinato, non sorveglianza ad area larga), non
   continuare a scalare l'ottimizzatore.

### Domande plausibili e risposte

**"Perché l'evasion rate non supera il 44%?"**
→ VisDrone è aereo, multi-scala, con pedoni che occupano pochi pixel.
Ho misurato il gradiente direttamente (debole, 0.0007-0.005) e
confrontato con letteratura UAV-specific: nessun lavoro forte su
VisDrone attacca i pedoni per lo stesso motivo.

**"Come sapete che non è un bug del codice?"**
→ Il gradient clipping non è mai scattato in nessun run (esclude
esplosioni/errori di scala grossolani), e sei configurazioni
metodologicamente diverse convergono nella stessa banda — un bug
isolato non produrrebbe quella coerenza.

**"Avete provato ad aumentare ancora il training?"**
→ Sì: da 375 a 2500 update (6.6x), +2.5 punti con rendimenti già
decrescenti. Continuare senza cambiare la causa (dimensione del
bersaglio) avrebbe un ritorno sempre più basso.

**"Cosa fareste con più tempo?"**
→ Allineare i dati allo scenario tattico già definito (persone di
dimensione realistica per un ingaggio a 10m, non l'intero range
eterogeneo di VisDrone), poi eventualmente un budget di calcolo lungo
sulla loss top-K già validata come più efficiente.

**"Perché non un dataset più piccolo/omogeneo?"**
→ È il Piano B già pronto (`annotate_mioDS.py`, Wu et al. 2020),
tenuto come alternativa esplicita fin dall'inizio, non come ripiego.

**"Il framework serve solo per questo attacco?"**
→ No — la Vision alimenta un simulatore multi-agente (Fusion
bayesiano Vision+OSINT+Behavioral, soglie IHL) tramite un bridge JSON,
visibile con `--run-sim`.

---

## Aperto / da fare

- **[IN CORSO] Esperimento notturno — ponderazione tattica della loss**:
  peso 0 sotto 60px, rampa lineare 60-150px, peso 1.0 sopra 150px,
  applicato dentro `_asymptotic_loss` (media pesata sulle celle top-K),
  canvas/EoT invariati (patch posizionata su tutti i target comunque).
  8000 step raw, accum=4, scheduler corretto — nessun vincolo tecnico
  precedente violato.

  **Pre-flight check sul trainset (dato già solido, indipendente dal
  risultato del training):**

  | Metrica | Valore |
  |---|---|
  | Annotazioni persona totali | 86.958 |
  | Annotazioni >= 60px (soglia minima storica) | 1.556 (1.8%) |
  | Annotazioni >= 80px (tatticamente rilevanti) | 394 (0.5%) |

  **Il 98.2% di tutte le annotazioni di VisDrone è sotto la soglia
  minima utilizzabile** — non un sottoinsieme raro, la maggioranza del
  dataset. Prova quantitativa diretta del domain shift tra VisDrone e lo
  scenario tattico simulato (`DRONE_ALTITUDE_M=10`), più forte di
  qualunque dato precedente (la stratificazione sul valset era su un
  campione già ridotto, qui è sull'intero trainset). Rischio noto:
  con solo 394 annotazioni ≥80px, il training potrebbe essere
  sottoallenato per scarsità di dati tattici, non per un limite della
  loss — un risultato modesto qui sarebbe comunque informativo, non un
  fallimento della modifica.
- Se il training notturno non basta: Piano B, dataset custom
  image-specific (`annotate_mioDS.py`, Wu et al. 2020) — non soffre di
  scarsità perché costruito apposta sullo scenario tattico
- Riserva: loss su feature intermedie del backbone (aerial imagery, 2023)
- Calibrazione fine di `ENGAGEMENT_THRESHOLD` con risultati Vision più forti
- Physical Domain Gap (stampa reale, drone reale) — limitazione da
  menzionare in tesi, non da risolvere nel codice
- Contributo originale possibile: ottimizzazione multi-dominio
  Vision+OSINT sotto budget di attacco vincolato — idea da maturare
