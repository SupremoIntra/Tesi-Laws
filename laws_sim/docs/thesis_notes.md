# Note di progetto — LAWS-SIM

Diario di lavoro: decisioni tecniche, esperimenti, risultati, errori e
correzioni. Ordinato in senso **cronologico-narrativo** (fasi), non per
categorie, perché la tesi deve raccontare un percorso di indagine: ipotesi →
misura → smentita → nuova ipotesi.

**Convenzioni del documento:**
- `[VERIFICATO]` = misurato, riproducibile con un comando documentato qui.
- `[IPOTESI]` = spiegazione plausibile **non** testata. Da non scrivere in
  tesi come fatto.
- `[RITIRATO]` = risultato ottenuto e poi respinto da un controllo di
  robustezza. Documentato deliberatamente: è metodo, non fallimento.
- `[APERTO]` = da decidere o da chiedere al relatore.

**Scope di questo documento**: Layer 1 (Vision — Universal Adversarial Patch
su YOLOv8n). Layer 2 (Simulation) e Layer 3 (Metadata Integrity / Fusion
Bayesiana OSINT) sono codice esistente ma **non ancora aggiornato** con i
risultati Okutama — trattati come fase successiva separata, non ancora
iniziata a fondo (vedi Parte V).

---

# Parte I — Il sistema

## Architettura: tre layer disaccoppiati

Collegati da file JSON, non da chiamate dirette:

1. **Vision** (`patch_optimizer.py`) — training di una Universal Adversarial
   Patch contro YOLOv8n, EoT completo (16 trasformazioni), loss asintotica.
2. **Simulation** (`simulator.py`, `entities.py`, `detection.py`) — ambiente
   multi-agente che legge il drop di F1 empirico da `vision_metrics.json`.
3. **Fusion/Decision** (`fusion_decision.py`) — sensor fusion bayesiano
   (Vision/OSINT/Behavioral) con soglie di ingaggio IHL, metrica CEAE/CLAE
   (**nome da disambiguare** — il codice importa `compute_clae`, il README
   descrive "CEAE, Cost-Effective Adversarial Engagement": verificare se sono
   la stessa metrica con sigla incoerente prima di scriverne in tesi).

Il ponte JSON è deliberato: i bug di visione non bloccano il simulatore
tattico, e permette esperimenti isolati (es. solo OSINT).

## Decisioni di design e loro giustificazione

Queste scelte sono **congelate**: tutti gli esperimenti riportati in questo
documento le condividono, ed è ciò che rende i confronti validi.

- **Loss asintotica** (`-log(1 - conf + eps)`) invece di Hinge Loss: la Hinge
  con soglia fissa ha derivata zero sotto soglia → gradiente morto. Con la
  Hinge, appena la confidenza scende sotto la soglia su una cella,
  l'ottimizzatore smette di ricevere segnale da quella cella anche se
  potrebbe spingerla ancora più giù — con centinaia di celle che scendono
  sotto soglia in momenti diversi, il training si blocca in un plateau
  ("vanishing gradient"). L'asintotica non ha soglia e non tocca mai zero:
  c'è sempre un minimo di spinta a scendere, elimina il problema
  strutturalmente.
- **Aggregazione top-K (K=20)** invece di media su tutte le celle mascherate:
  generalizza la tecnica "max objectness" di Thys et al. (2019) — dentro la
  maschera spaziale del bersaglio ci sono centinaia di celle, la maggior
  parte sfondo con confidenza già vicina a zero; mediare su tutte diluisce il
  segnale utile. Concentrandolo sulle 20 celle più confidenti (quelle dove
  YOLO "guarda" davvero il centro della persona) si ottiene lo stesso
  risultato finale con metà del calcolo. Scelta poi **validata empiricamente**
  (Fase 4).
- **TV_WEIGHT = 0.1**: un TV loss basso produce rumore ad alta frequenza che
  l'interpolazione bilineare di `grid_sample` nell'EoT distrugge (l'EoT
  applica una media spaziale dei pixel — un filtro passa-basso — quindi
  qualunque texture ad alta frequenza viene semplicemente cancellata prima
  che la loss possa usarla). Un peso alto forza pattern a bassa frequenza
  (macchie di colore ampie) che sopravvivono alla trasformazione, e sono
  anche più facili da stampare fisicamente in modo fedele.
- **Parametrizzazione sigmoide** dei pixel patch (non `clamp`): clamp uccide
  il gradiente ai pixel che toccano 0 o 1 (derivata zero al bordo); sigmoide
  mantiene un gradiente fluido ovunque, al prezzo di un rischio di
  saturazione se il learning rate è troppo alto.
- **PATCH_BBOX_COVERAGE = 0.20**: coerente con `get_chest_bbox_proportional`
  (50% larghezza × 40% altezza = 0.20 dell'area del bersaglio, "Tactical Vest
  Assumption" — copre il petto, la zona più visibile da prospettiva drone).
- **N_TARGETS=3, N_CIVILIANS=30**: rapporto 1:10, scenario "Urban Clutter"
  (Arkin 2009) per stressare il Fusion Agent sotto OSINT poisoning verso falsi
  positivi sui civili.
- **ENGAGEMENT_THRESHOLD=0.58 / ALERT=0.38 / TRACK=0.22**: la soglia di
  ingaggio volutamente vicina al 50% modella l'abbassamento degli standard di
  certezza operativa osservato in scenari reali (Automation Bias).
- **Split train/val separati, su entrambi i dataset**: allenare e valutare
  sulle stesse immagini avrebbe inflazionato l'evasion rate — la patch
  "impara a memoria" quelle scene specifiche invece di generalizzare. Su
  Okutama il vincolo è più forte — mai lo stesso *video*, non solo mai la
  stessa immagine (frame dello stesso video sono quasi identici tra loro).
- **Nessun modello analitico di decadimento distanza-based**: il degrado
  visivo è solo data-driven, dal drop di F1 empirico.

## Glossario

- **Step raw**: un ciclo del training loop (immagine → EoT → forward →
  backward). `PATCH_STEPS` conta questi, non gli aggiornamenti dei pixel.
- **Gradient Accumulation**: quanti step raw si sommano prima di un update
  reale. `update_reali = step_raw / accumulation_steps`. Con N=4 servono 4
  immagini prima che i pixel della patch cambino davvero una volta —
  simula un batch più grande senza il costo di memoria di caricarlo tutto
  insieme, al prezzo di meno aggiornamenti reali a parità di immagini viste.
- **Update reale**: la chiamata a `optimizer.step()` — il momento in cui i
  pixel della patch cambiano davvero.
- **EoT**: media della loss su N trasformazioni random della patch (rotazione,
  scala, colore), per farla funzionare su una distribuzione di condizioni
  realistiche, non su un'immagine esatta — una patch fisica reale viene vista
  dal drone con angolazioni/distanze/luce diverse ogni volta.
- **Evasion Rate** (= 1 − R1): frazione di frame con persona reale in cui YOLO
  non rileva nessuno.
- **R1 / Sensitività**: frazione di frame positivi in cui YOLO rileva.
- **R2 / Specificità**: frazione di frame negativi in cui YOLO non allucina.
- **√(R1·R2)**: media geometrica, metrica **primaria** indicata dal relatore.
  Il prodotto di due valori in [0,1] è sempre ≤ del più piccolo dei due: per
  farla salire devono salire ENTRAMBI i recall, e devono essere vicini tra
  loro — impedisce di "barare" gonfiando un recall a scapito dell'altro.
- **F1**: metrica **secondaria** ("la usano tutti ma fa caos").
- **Frame positivo**: contiene ≥1 bersaglio ≥60px di altezza bbox.
- **Frame ambiguo**: contiene persone, ma tutte <60px. Trattato come negativo
  con convenzione ignore-region (Fase 4).
- **Danno collaterale**: detection spurie **dentro i frame positivi**, dove la
  patch è realmente presente. Metrica introdotta per rispondere alla domanda
  che R2 per costruzione non può testare (Fase 5).
- **Norma del gradiente / gradient clipping**: la norma è un singolo numero
  che misura quanto è "grande" il vettore gradiente nel complesso — quanto è
  forte il segnale di aggiornamento in un dato istante. Il clipping è una
  rete di sicurezza: se la norma supera 1.0, viene ridimensionata per evitare
  update troppo bruschi. Non è mai scattata nei nostri esperimenti — segno
  che il problema non è mai stato un gradiente troppo grande, ma uno troppo
  debole.

## Glossario dei tool (`tools/`)

Aggiunto perché con due dataset e più run il numero di script cresce e i nomi
da soli non bastano a ricordare cosa fa cosa.

| Tool | Cosa fa | Quando usarlo |
|---|---|---|
| `count_negative_candidates.py` | Conta frame positivi/negativi/ambigui | Prima di allenare, verifica campione |
| `stratify_by_size.py` | Evasion rate per fascia di altezza bbox, con una patch già addestrata | Diagnosi soffitto strutturale/scala |
| `plot_k_selection.py` | Curve F1/√(R1·R2)/R1/R2 al variare di K (nessuna patch, comportamento naturale) | Scelta/verifica di `LOSS_TOP_K` |
| `bootstrap_ci_report.py` | CI bootstrap PRE/POST, versione precedente (solo F1+gmean, no R1/R2 separate) | Superato da `--eval-report`, tenuto per lo storico di Fase 6 |
| `plot_CI_box.py` | Grafico a candela (punto+baffi CI95%) da `full_report_*.json` | Figura per la tesi, dopo `--eval-report` |
| `plot_training_curves.py` | Curva Loss/TV Loss durante un training, da `training_metrics_*.json` | Dopo `--train-patch`, se serve il grafico |
| `plot_runs_comparison.py` | Barre di confronto tra le 6 configurazioni storiche (lista `RUNS` scritta a mano nello script) | Solo per il confronto Fase 1-2, statico, non dipende da file generati |
| `generate_before_after_images.py` | Immagini affiancate PRE/POST con box disegnate, crop leggibile | Figure per slide/tesi, prova visiva oltre alle tabelle |
| `annotate_mioDS.py` | "Piano B": auto-annotazione foto raw in formato VisDrone (Wu et al. 2020) | Non usato — solo citato come alternativa di design in "Lavoro futuro" |

## Convenzione di naming degli output (adottata in Fase 7, tardi ma adottata)

Problema che ha causato confusione reale in sessione: molti tool scrivevano
su path fissi (`full_report.json`, `k_selection_raw.npz`,
`training_metrics.json`) senza indicare dataset/config nel nome — un run
successivo sovrascriveva silenziosamente quello precedente, e file di run
diversi (es. Fase 6 VisDrone vs Okutama) sono stati scambiati per errore.

**Convenzione adottata**: `<nome_base>_<loader>[_<img_size>][_stride<N>].ext`,
stessa cartella `outputs/metrics/` (nessuna sottocartella, minima
ristrutturazione). Esempio: `full_report_okutama_960_stride27.json`. I file
storici con nome generico (`full_report.json`, `bootstrap_ci_report.json` di
Fase 6) sono stati spostati in `outputs/metrics/archive_fase6_visdrone/`.

**Perdita nota, irreversibile**: `training_metrics.json` (path fisso prima
del fix) è stato sovrascritto più volte — la curva di loss del run VisDrone
finale (Fase 2, config #5) non è più rigenerabile. Non impatta i risultati
numerici (già consolidati come tabelle), solo un grafico di supporto. Da ora
ogni `--train-patch` salva una copia con suffisso.

---

# Parte II — Diario di viaggio

## Fase 0 — Far funzionare la pipeline su Apple Silicon

Hardware: MacBook Air M4, 16GB di memoria **unificata** (CPU e GPU la
condividono — vincolo che tornerà decisivo in Fase 7).

Bug risolti in questa fase, tutti di infrastruttura:

| # | Problema | Causa | Fix |
|---|---|---|---|
| 1 | `grid_sampler_2d_backward` non implementato su MPS | EoT usa `F.grid_sample` per rotazione/scala | `PYTORCH_ENABLE_MPS_FALLBACK=1`: solo quell'operatore va su CPU, il resto resta su MPS |
| 2 | `RuntimeError: view size is not compatible...` nel primo conv2d di YOLO | `.expand()` + `.permute()` lasciano un tensore non contiguo, MPS non lo tollera | `.contiguous()` dopo il permute dell'immagine e sul batch finale |
| 3 | `config.py` senza `N_TARGETS`, `FUSION_WEIGHTS`, soglie | Costanti perse in un refactor precedente | Ricostruite con fonte citata per ognuna |
| 4 | Nessuna detection nel simulatore (TP/FP sempre 0) | `DRONE_ALTITUDE_M=300` sempre oltre `YOLO_MAX_RANGE=150` | Altitudine realistica: 10m |
| 5 | Patch copriva l'intera figura (incluso volto) | `PATCH_H/W` fissi in pixel assoluti | `get_chest_bbox_proportional`: patch scalata sul bbox reale |
| 6 | Vanishing gradient con Hinge Loss | derivata zero sotto soglia | Sostituita con loss asintotica |
| 7 | LR resta al massimo per quasi tutto il run invece di scendere con la curva coseno | `CosineAnnealingLR(T_max=n_steps)` usa step raw, ma `scheduler.step()` è chiamato una volta per **update reale** — con accum=4 la curva si allunga 4× | `T_max = n_steps // GRADIENT_ACCUMULATION_STEPS` |

Il bug #7 è metodologicamente importante: **tutti gli esperimenti prima della
sua scoperta usavano uno scheduler allungato**, quindi i confronti nella
cronologia sottostante indicano esplicitamente quali run avevano lo scheduler
corretto e quali no.

## Fase 1–2 — Sei configurazioni e la scoperta del soffitto

`[VERIFICATO]` VisDrone, 640×640.

| # | Config | Update reali | Scheduler | F1 | Evasion Rate |
|---|---|---|---|---|---|
| 0 | Baseline storico (Hinge Loss) | 3000 | corretto | 0.760 | 38.7% |
| 1 | Loss asintotica, mean, accum=4 | 375 | allungato 4× | 0.740 | 41.25% |
| 2 | Loss asintotica, mean, accum=2 | 750 | allungato 2× | 0.750 | 40.0% |
| 3 | Loss asintotica, mean, accum=4 | 2500 | corretto | 0.720 | **43.75%** |
| 4 | Loss asintotica, mean, accum=16 | 625 | corretto | 0.740 | 41.25% |
| 5 | **Loss asintotica, top-K=20, accum=4** | **1250** | **corretto** | **0.720** | **43.75%** |

**Riga 5 è il riferimento**: stesso massimo della riga 3 con metà del budget di
calcolo — la configurazione più efficiente misurata, ed è essa stessa la prova
del punto seguente.

### Il finding: soffitto strutturale, non limite di ottimizzazione

Sei configurazioni radicalmente diverse — due formule di loss, due aggregazioni
(mean/top-K), accumulo da 1 a 16, scheduler corretto o meno — convergono nella
stessa banda stretta (38.7%–43.75%). Se il limite fosse "serve la loss giusta"
o "servono più update", almeno una configurazione l'avrebbe superato nettamente.

`[VERIFICATO]` **Diagnosi del gradiente, misurata direttamente:** la norma del
gradiente per immagine singola è debole (0.0007–0.005); il gradient clipping
(soglia 1.0) non è mai scattato in nessun run. Aumentare l'accumulo da 4 a 16
ha **peggiorato** il risultato: prova che il segnale utile condiviso tra scene
diverse è intrinsecamente debole, non un problema di quantità di update.

`[VERIFICATO]` **Conferma dalla letteratura UAV-specific** (non dai paper
generici a livello del suolo): nessun lavoro con risultati forti su VisDrone
attacca i pedoni.

| Paper | Anno | Bersaglio su VisDrone | Perché |
|---|---|---|---|
| Shrestha, Pathak, Viegas — "Towards a Robust Adversarial Patch Attack Against UAV Object Detection" | 2023 | Car (80% ASR white-box) | Patch pensate per prospettiva/distanza UAV |
| LFRAP (Multi-Dimensional Feature Optimization) | 2025 | Car, truck, van, bus — pedoni esplicitamente scartati | Dimensione ridotta dei pedoni in prospettiva overhead |
| Adversarial patch attacks against aerial imagery object detectors | 2023 | Aerei, loss su feature intermedie | Bersagli grandi + tecnica feature-level |

Il soffitto al 44% è verosimilmente il prezzo di aver scelto il bersaglio più
difficile del dominio (persone, non veicoli), non un limite del framework.

### Prima stratificazione per dimensione (VisDrone, 640×640)

`[VERIFICATO]` `tools/stratify_by_size.py`, patch VisDrone già addestrata:

| Bucket altezza | Evasi/Totali | Evasion Rate |
|---|---|---|
| 60–100px | 34/314 | 10.8% |
| 100–150px | 1/56 | 1.8% |
| 150px+ | 5/11 | **45.5%** |

Pattern **non monotono**, e il bucket promettente (150px+) ha **11 casi
totali** — statisticamente inutilizzabile. Trattato all'epoca come pista
esplorativa, non risultato. *Questa domanda aperta è quella che Okutama
risolverà in Fase 7.*

`[IPOTESI]` due effetti opposti: (1) budget di pixel — bersagli grandi danno
più superficie all'attacco; (2) margine di confidenza di partenza — bersagli
piccoli hanno già confidenza bassa e basta poco per spingerli sotto soglia,
bersagli medi ben risolti hanno confidenza alta che una patch debole non
scalfisce.

## Fase 3 — Scelta empirica di K (validazione a posteriori di un'assunzione)

`[VERIFICATO]` `tools/plot_k_selection.py`, valset VisDrone completo (531
frame: 80 positivi / 451 negativi).

**Il problema:** `LOSS_TOP_K=20` era un'assunzione iniziale, non una scelta
misurata. Il relatore ha chiesto un metodo per giustificarla o correggerla.

**Metodo:** per ogni frame positivo (senza patch — si caratterizza il
comportamento *naturale* di YOLO) si ordinano le 8400 celle pre-NMS per
confidenza decrescente. Con una somma cumulativa si ottengono TP/FP/TN/FN a
livello di cella per **ogni** K da 1 a 8400 senza rifare l'inferenza: il costo
è dominato dalla singola passata YOLO, non dal ciclo su K.

| Criterio | K ottimo | Cosa penalizza |
|---|---|---|
| F1 | **244** (plateau da K≈150) | copertura geometrica incompleta del bersaglio |
| Media geometrica √(R1·R2) | **37** | diluizione nello sfondo (R2 crolla rapidamente con K) |

**Perché i due criteri divergono (questo è il finding, non il numero):** dentro
l'impronta geometrica di un bersaglio (~200–250 celle, tre stride YOLO) solo un
nucleo ristretto (~30–40 celle) ha confidenza realmente alta; il resto è
periferia dentro il bbox con segnale debole. F1 premia la copertura totale
(K grande); la media geometrica penalizza la diluizione appena K supera il
nucleo reale.

**Decisione:** `LOSS_TOP_K=20` è vicino a K=37 (media geometrica, criterio
**primario** del relatore), non a K=244 (F1, secondario). Non era
un'assunzione fortunata: era già dentro il nucleo di confidenza reale.
**Conseguenza: nessun nuovo training.** K=244 resta annotato come lavoro
futuro (una loss che copra l'intera impronta geometrica), non come azione.

**Definizione di R2 nei plot vs K — risolta.** R2 è calcolata a livello di
**cella**, con soglia implicita tau(K) = valore di confidenza in posizione K
sui frame positivi, applicata ai negativi. Necessario: a livello di frame con
soglia fissa il grafico "R2 vs K" sarebbe una riga piatta, inutile per
scegliere K. La lettura cell-level produce il trade-off monotono che rende il
picco della media geometrica un criterio reale.

## Fase 4 — Le metriche del relatore, e un errore scoperto strada facendo

`[VERIFICATO]` `src/metrics.py`, `src/simulator.py:evaluate_on_dataset`,
`cli.py --eval-report`.

### Metodo statistico

1. **CI indipendenti** — bootstrap percentile (10.000 iterazioni, 95%) su PRE
   e POST separatamente. Corretto ma conservativo.
2. **Bootstrap appaiato del delta** — PRE e POST sono valutati sugli *stessi*
   frame, quindi sono misure appaiate. Si ricampiona una sola lista di indici
   per iterazione e la si applica a entrambe le condizioni, calcolando
   Δ = POST − PRE. Sfrutta la correlazione intra-frame, dà stima più precisa e
   p-value a due code. Riferimento: Efron & Tibshirani (1993).

### Il problema del campione negativo, e la correzione

`[VERIFICATO]` `tools/count_negative_candidates.py` su VisDrone-val (n=531):

| Categoria | n | % |
|---|---|---|
| Positivo (≥1 bersaglio ≥60px) | 80 | 15.1% |
| Negativo vero (0 persone, qualunque size) | 9 | 1.7% |
| Ambiguo (solo persone <60px) | 442 | 83.2% |

Con soli 9 negativi veri, R2 era una stima degenere. **Correzione:** i 442
frame ambigui diventano negativi a pieno titolo, con matching IoU contro
*tutte* le persone annotate (anche <60px): IoU ≥ 0.3 con una persona di
qualunque dimensione → detection corretta su bersaglio non tatticamente valido
→ **ignorata** (né TP né FP, convenzione ignore-region); nessun match →
**falso positivo vero**. Soglia IoU=0.3 permissiva rispetto allo standard
PASCAL VOC 0.5 (Everingham et al. 2010), motivata dalla sensibilità dell'IoU su
bbox piccoli (Yu et al. 2020) e allineata a CrowdHuman (Shao et al. 2018).

### Risultato consolidato VisDrone

`[VERIFICATO]` n=531 (80 pos + 451 neg), conf=0.5, 10.000 iter, bootstrap
appaiato:

| Metrica | PRE | POST | Δ [CI 95%] | p | Signif. |
|---|---|---|---|---|---|
| Evasion rate (1−R1) | 0.1750 [0.096, 0.264] | 0.4250 [0.317, 0.533] | +0.250 [+0.156, +0.346] | <0.0001 | **Sì** |
| Sensitività R1 | 0.8250 [0.736, 0.904] | 0.5750 [0.467, 0.683] | −0.250 [−0.346, −0.156] | <0.0001 | **Sì** |
| Specificità R2 | 0.9690 [0.952, 0.984] | 0.9690 [0.952, 0.984] | 0.000 | 1.0000 | No (vedi sotto) |
| √(R1·R2) | 0.8941 [0.844, 0.937] | 0.7464 [0.672, 0.814] | −0.148 [−0.210, −0.090] | <0.0001 | **Sì** |
| F1 (secondaria) | 0.8250 [0.755, 0.883] | 0.6571 [0.561, 0.744] | −0.168 [−0.242, −0.100] | <0.0001 | **Sì** |

### ⚠ Il finding più importante della fase: R2 è tautologico per costruzione

`[VERIFICATO]` Con n=531, R2 PRE e POST risultano identici **fino alla
sedicesima cifra decimale** (0.9689578713968958 in entrambi i casi).

Causa: la patch è disegnata solo dentro il ciclo `for bbox in
valid_gt_bboxes`, che itera **zero volte** quando il frame non ha bersagli
≥60px. Per tutti i frame negativi l'immagine passata a YOLO è identica con o
senza patch, e YOLO in inferenza è deterministico.

**Espandere il campione da 9 a 451 non ha reso R2 più informativo sull'effetto
della patch — l'ha reso più preciso su una quantità che per costruzione non
può muoversi.** Il vecchio risultato "R2=1.0 invariato" con n=9 non era la
dimostrazione di un attacco pulito: era la stessa tautologia, solo meno
visibile con un valore rotondo.

**Reinterpretazione da usare in tesi:** R2=0.9690 è una **caratterizzazione del
detector** — tasso di falso allarme di base di YOLOv8n su sfondi VisDrone privi
di bersaglio valido (~3.1%) — non un test dell'attacco. Conseguenza diretta:
**tutto il movimento di √(R1·R2) è attribuibile a R1**; R2 agisce da
moltiplicatore costante, non da secondo asse indipendente.

### Robustezza multi-soglia

`[VERIFICATO]` Evasion, R1, √(R1·R2), F1 restano significativi (p tra 0.0002 e
<0.0001) a conf = 0.3 / 0.5 / 0.7. R2 identico PRE/POST a **tutte** le soglie
(0.8248, 0.9690, 1.0000) — conferma indipendente che l'invarianza è strutturale,
non una coincidenza numerica.

**Scoperta laterale:** l'evasion rate PRE (senza attacco) passa da 3.75%
(soglia 0.3) a 72.5% (soglia 0.7) — conferma indipendente del soffitto
strutturale e giustifica retroattivamente 0.50 come default ragionevole.

**Metrica ritirata:** un tool di "confidence drop" (calo continuo per frame) è
stato rimosso su indicazione del relatore (preferito il prima/dopo binario) e
perché misurava il massimo per-frame, confrontando potenzialmente detection
diverse tra PRE e POST.

## Fase 5 — Danno collaterale su VisDrone `[RITIRATO]`

**Perché introdotta:** R2 non può testare l'attacco (Fase 4). Questa metrica
misura invece, **dentro** i frame positivi dove la patch è realmente presente,
se l'attacco introduce detection spurie senza corrispondenza IoU con nessuna
persona reale. Domanda diversa e genuinamente testabile — non un tentativo di
"salvare" R2.

**Metodo:** conteggio grezzo di allucinazioni per frame (non indicatore
binario) — più potenza statistica sugli stessi 80 frame positivi, a costo zero.
Una prima versione a indicatore binario (PRE 8/80, POST 4/80, p=0.0374) è stata
superata da questa e non va citata.

`[VERIFICATO]` Verifica di robustezza, richiesta **prima** di accettare il
risultato:

| Soglia | PRE | POST | Δ | p | Signif. |
|---|---|---|---|---|---|
| 0.3 | 0.4625 | 0.4375 | −0.0250 | 0.535 | No |
| 0.5 | 0.1125 | 0.0500 | −0.0625 | 0.0136 | Sì |
| 0.7 | 0.0000 | 0.0000 | 0.0000 | 1.000 | No (nessun segnale in entrambe le condizioni) |

**Esito: il pattern non replica.** Significativo solo esattamente a 0.50; a 0.3
stessa direzione ma non significativo; a 0.7 segnale nullo. Inoltre p=0.0136
non sopravvive a Bonferroni per 6 confronti (richiesto p<0.0083).

**Conclusione: ritirato come risultato positivo.** Documentato come esempio di
verifica metodologica corretta — un segnale debole è stato messo in dubbio,
testato, e respinto quando il controllo non lo confermava. *Un risultato
negativo verificato con rigore vale più di un risultato positivo non
controllato.*

## Fase 6 — Ponderazione tattica della loss `[RITIRATO]`

Ipotesi: se il problema è la scarsità di bersagli grandi, pesare la loss verso
di essi dovrebbe aiutare. Implementazione: peso 0 sotto 60px, rampa lineare
60–150px, peso 1.0 sopra 150px, dentro `_asymptotic_loss`. Canvas/EoT
invariati. 8000 step raw, accum=4, scheduler corretto.

`[VERIFICATO]` **Pre-flight sul trainset VisDrone** — dato solido e
indipendente dall'esito del training:

| Metrica | Valore |
|---|---|
| Annotazioni persona totali | 86.958 |
| Annotazioni ≥60px (soglia minima storica) | 1.556 (**1.8%**) |
| Annotazioni ≥80px (tatticamente rilevanti) | 394 (**0.5%**) |

`[VERIFICATO]` **Risultato:**

| Metrica | Valore |
|---|---|
| F1 (completo) | 0.730 |
| Evasion Rate (completo) | 42.5% |
| Evasion Rate (target ≥80px) | 41.7% |
| Copertura tattica valset (≥80px) | 32.0%* |

\* denominatore diverso dal pre-flight (0.5%): qui è % tra le annotazioni già
filtrate ≥60px nel **valset**, non sul totale grezzo del **trainset**. Le due
percentuali misurano cose diverse.

**Nessun salto.** Risultati indistinguibili da Fase 2 (43.75%); l'evasion
filtrata (41.7%) è persino più bassa di quella completa (42.5%) — l'opposto
dell'ipotesi. Con solo 394 annotazioni ≥80px in tutto il trainset, la scarsità
di dati ha vinto sulla ponderazione. Segnale indiretto a supporto: `YOLO Conf`
nel log resta alto (0.35–0.76) invece di crollare — la loss *si sta* misurando
con i bersagli grandi, ma senza abbastanza esempi per imparare a batterli.

**Conclusione operativa:** la strada non è il loss-reweighting ma un dataset
con distribuzione di scala diversa. Il pre-flight (98.2% delle annotazioni
VisDrone sotto 60px) resta la prova quantitativa che regge indipendentemente
dall'esito del training.

**Nota per un eventuale retest su Okutama** (ipotesi discussa, non ancora
eseguita in modo conclusivo in questa sessione): il fattore limitante qui
identificato — troppo pochi bersagli grandi nel trainset — è quantitativamente
diverso su Okutama (il bucket 100–150px ha 598 casi nel solo valset, contro
394 ≥80px in tutto il trainset VisDrone). Se un retest è stato lanciato e
concluso, il suo risultato va aggiunto qui con lo stesso rigore (pre-flight +
verifica di robustezza) prima di scriverlo in tesi.

**VisDrone si chiude qui.** Pipeline validata, numeri definitivi, non più
toccata.

---

## Fase 7 — Migrazione a Okutama-Action

Richiesta del relatore: testare un dataset con pedoni ripresi più da vicino
**separatamente**, e confrontare. Non accorpare, non sostituire. Se
statisticamente non migliora, è comunque un risultato valido.

Candidato scelto: **Okutama-Action** (Barekatain et al., CVPR-W 2017;
altitudine 10–45m, coerente con `DRONE_ALTITUDE_M=10`; CC BY-NC-SA). Dataset
SAR (HERIDAL, SARD, LADD) scartati: stesso problema di oggetti piccoli.

### 7.1 Setup, e tre decisioni prese esplicitamente

- **Dati:** frame pre-estratti a 1280×720 (train 5.3GB, test 1.5GB) — nessun
  decoding dei video 4K.
- **Label:** `SingleActionLabels/3840x2160/`. Una riga per persona per frame;
  le colonne azione si ignorano (istruzione esplicita della documentazione
  ufficiale per il task di pedestrian detection, non una nostra
  semplificazione). Coordinate in spazio **nativo 3840×2160** — non 1280×720:
  errore facile e silenzioso, gestito con rescale esplicito in un solo
  passaggio (nessun passaggio intermedio per 1280×720, che serve solo a
  leggere i pixel del frame).
- **Split:** ufficiale del dataset. Le label del test set **sono pubbliche**
  (verificato sulla pagina ufficiale; una fonte secondaria del 2018 sosteneva
  il contrario e si è rivelata superata). Nessuno split manuale necessario —
  lo split ufficiale è già per scenario, quindi rispetta il criterio "mai lo
  stesso video in train e in eval".
- **Loader:** `src/okutama_loader.py`, interfaccia identica a `VisDroneLoader`
  (`get_sample`, `__len__`, `iter_batches`) → il resto della pipeline funziona
  senza modifiche.

### 7.2 Il vincolo hardware, e perché la risoluzione è 960 e non 1280

Decisione iniziale: 1280×1280, come compromesso tra preservare la scala dei
bersagli e mantenere parità metodologica di preprocessing con VisDrone.

`[VERIFICATO]` **1280 non è eseguibile su questo hardware.** Con EoT=16 e
float32 il processo raggiunge **~17 GB residenti su 16 GB totali** →
swap continuo (contatore `swapouts` in crescita costante, processo in stato
`stuck`, throughput crollato: fermo allo step 40 dopo un'ora). Non
"lento": non completabile.

`[VERIFICATO]` **960×960 rientra nel budget:** ~12 GB residenti, delta
`swapouts` = 0, processo `running`, 8000 step completati.

**640×640 (parità esatta con VisDrone) scartato con un calcolo, non a
sensazione:** a 640 ogni altezza bbox si dimezza rispetto a 1280, quindi una
bbox resta ≥60px solo se era ≥120px a 1280. Dalla distribuzione misurata a
1280 questo cancella l'intero bucket 60–100px (87.9% del totale): le bbox
valide crollerebbero da 46.293 a ~2.400, **−95%**, riportando il problema che
la migrazione doveva risolvere.

`[VERIFICATO]` **Costo reale della scelta 960:** bbox valide ≥60px scese da
46.293 (@1280) a 20.539 (@960), **−56%**. Da dichiarare come limitazione, non
da nascondere.

### 7.3 Pre-flight (costo zero, prima di allenare)

`[VERIFICATO]` `count_negative_candidates.py --loader okutama --img-size 960`
su `okutama_val`, n=14210 — **misura corretta e definitiva, a risoluzione
coerente con training/eval** (una prima misura fatta a 1280, prima
dell'aggiunta di `--img-size` al tool, è stata sostituita da questa):

| Categoria | n | % |
|---|---|---|
| Positivo (≥1 bersaglio ≥60px) | 8.095 | **57.0%** |
| Negativo vero (0 persone) | 0 | 0.0% * |
| Ambiguo (solo persone <60px) | 6.115 | 43.0% |

\* Limite noto del loader: `_build_index` indicizza solo frame con ≥1
annotazione, quindi i frame a zero persone non sono rappresentati. Non
bloccante — la classe negativa estesa usa positivi+ambigui, e 6.115 ambigui
superano ampiamente i 451 di VisDrone. Da dichiarare, non da correggere.

**Confronto diretto con VisDrone: 57.0% vs 15.1% di frame positivi.** È la
giustificazione quantitativa della migrazione: non un'ipotesi, una misura —
quasi 4× più frame utilizzabili, anche dopo aver scontato il costo della
risoluzione ridotta.

### 7.4 Training

`[VERIFICATO]` Comando:

```bash
python cli.py --train-patch --loader okutama --train-dir data/okutama_train \
  --patch-out outputs/patches/patch_okutama.pt --img-size 960 --fresh
```

Config **invariata** rispetto a VisDrone: EoT=16, accum=4 (batch effettivo 4),
top-K=20, TV_WEIGHT=0.1, PATCH_LR=0.01, 8000 step. Trainset: 54.664 frame
validi. Esito: completato, nessun crash, nessuno swap.

**Osservazione dal log:** TV Loss in discesa monotona (0.0628 → 0.0466) — la
patch si smootha regolarmente. Loss totale oscillante 0.70–0.85 senza trend
netto. Ultimo checkpoint "best" a step ~5973: **nessun miglioramento negli
ultimi ~2000 step**, quindi plateau raggiunto prima della fine (coerente con
Brown et al. 2017 sui rendimenti marginali). L'early stopping a pazienza 200
non è scattato.

### 7.5 Il problema della correlazione temporale, e come è stato risolto

Okutama è video a ~30fps: i frame consecutivi sono quasi identici (verificato a
occhio sui sample 0/1/2 — bbox che differiscono di 1–2 pixel). Il bootstrap
appaiato tratta i frame come **indipendenti**: vero per le 531 immagini
VisDrone, **falso** su un dataset video.

Conseguenza: con n=14210 i CI95% sono artificialmente stretti e i p-value
sovrastimati — pseudo-replicazione. Le *stime puntuali* restano valide; è
l'incertezza a essere sottostimata.

**Soluzione:** aggiunto `--stride` a `--eval-report` (sottocampionamento 1 frame
ogni N). Stride=27 → 527 frame, separati da ~0.9s, numericamente comparabili al
n=531 di VisDrone.

`[VERIFICATO]` **La decorrelazione non cambia le conclusioni, corregge
l'incertezza:**

| Metrica | n=14210 (correlato) | n=527 (stride 27) |
|---|---|---|
| Evasion PRE | 0.4988 | 0.4967 |
| Evasion POST | 0.7800 | 0.7767 |
| Δ Evasion | +0.2812 [+0.2712, +0.2909] | +0.2800 [+0.2281, +0.3322] |

Stima puntuale praticamente identica, **CI ~5× più largo** — esattamente
l'effetto atteso rimuovendo la falsa indipendenza. Tutto resta significativo
(p<0.0001). La conclusione non era un artefatto della pseudo-replicazione.

**Il dato da citare in tesi è quello a n=527.**

### 7.6 Risultato consolidato Okutama

`[VERIFICATO]` n=527 (stride 27), 960×960, conf=0.5, 10.000 iter, bootstrap
appaiato:

| Metrica | PRE | POST | Δ [CI 95%] | p | Signif. |
|---|---|---|---|---|---|
| Evasion rate (1−R1) | 0.4967 [0.4406, 0.5531] | 0.7767 [0.7290, 0.8223] | +0.2800 [+0.2281, +0.3322] | <0.0001 | **Sì** |
| Sensitività R1 | 0.5033 [0.4469, 0.5594] | 0.2233 [0.1777, 0.2710] | −0.2800 [−0.3322, −0.2281] | <0.0001 | **Sì** |
| Specificità R2 | 0.9604 [0.9331, 0.9830] | 0.9604 [0.9331, 0.9830] | 0.0000 | 1.0000 | No (invariante per costruzione) |
| √(R1·R2) | 0.6953 [0.6539, 0.7340] | 0.4631 [0.4125, 0.5103] | −0.2321 [−0.2783, −0.1881] | <0.0001 | **Sì** |
| F1 (secondaria) | 0.6565 [0.6044, 0.7054] | 0.3564 [0.2936, 0.4169] | −0.3001 [−0.3578, −0.2440] | <0.0001 | **Sì** |

**R2 invariante anche qui** — la stessa proprietà strutturale di Fase 4 si
replica su un dataset indipendente. Conferma che è una caratteristica del
disegno sperimentale, non un artefatto di VisDrone.

### 7.7 Danno collaterale su Okutama — risultato **nuovo**, non replica

`[VERIFICATO]` Frame positivi decorrelati, n=300:

| Soglia | PRE | POST | Δ | p | Esito |
|---|---|---|---|---|---|
| 0.3 | 0.3700 [0.2933, 0.4500] | 0.2867 [0.2200, 0.3567] | −0.0833 [−0.1500, −0.0200] | 0.0108 | **Significativo** |
| 0.5 | 0.0967 [0.0633, 0.1333] | 0.0433 [0.0200, 0.0700] | −0.0533 [−0.0833, −0.0267] | <0.0001 | **Significativo** |
| 0.7 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | Non misurabile (zero eventi) |

Controprova sul campione pieno `[VERIFICATO]` (n=8095 positivi, correlato — CI
non affidabili, ma utile per la *potenza*): 0.3 → Δ−0.1138 p<0.0001; 0.5 →
Δ−0.0378 p<0.0001; 0.7 → Δ−0.0014 **p=0.0038**.

**Lettura onesta — questo caso è diverso da VisDrone.** Su VisDrone il pattern
non replicava *in direzione* (a 0.3 il calo svaniva): rumore. Qui il pattern è
**monotono e coerente su entrambi i campioni**: l'effetto si assottiglia man
mano che la soglia sale (−0.11 → −0.05 → ~0), fino a diventare un evento così
raro che con 300 frame **non capita mai, né PRE né POST** — CI degenere per
assenza di dati, non per assenza di effetto. Sul campione pieno lo stesso
effetto minuscolo resta misurabile (p=0.0038) perché la potenza è sufficiente.

**Formulazione da usare in tesi:** robusto a 0.3 e 0.5; a 0.7 l'effetto è
presente e nella stessa direzione ma **sotto la soglia di rilevabilità** con
questo campione — "non misurabile", non "assente". Il segno è **negativo**: la
patch produce *meno* detection spurie, non più (stessa direzione osservata su
VisDrone). Riportare la tabella completa a tre soglie: il solo valore a 0.5
sovrastima l'effetto.

### 7.8 Stratificazione per dimensione — la domanda di Fase 2 trova risposta

`[VERIFICATO]` `stratify_by_size.py`, patch **addestrata su Okutama**, 960×960:

| Bucket altezza (@960) | Evasi/Totali | Evasion Rate |
|---|---|---|
| 60–100px | 9155/19921 | 46.0% |
| 100–150px | 598/618 | **96.8%** |
| 150px+ | 0/0 | bucket vuoto |

Il bucket 150px+ vuoto è **atteso, non un bug**: a 960 le altezze sono 0.75×
quelle a 1280, quindi ≥150px@960 ≡ ≥200px@1280, e a 1280 c'erano solo 316 bbox
≥150px in totale.

**Controllo di coerenza indipendente** `[VERIFICATO]`: la stratificazione fatta
in precedenza a 1280 con la patch **VisDrone** dava 60–100px: 52.2%
(n=40683), 100–150px: 48.6% (n=5294), 150px+: **97.2%** (n=316). Convertendo le
scale (960 → 1280 = ×1.333), il bucket 100–150@960 corrisponde a 133–200@1280 —
cioè proprio la regione dove la misura a 1280 mostrava il salto al 97%. **Due
misure indipendenti, patch diverse e risoluzioni diverse, individuano il salto
nella stessa regione di scala.**

**Cosa risolve:** la domanda lasciata aperta in Fase 2 (bucket 150px+
promettente ma con soli 11 casi) ora ha n=618 con effetto molto ampio. La
direzione dell'ipotesi di scala è confermata su grande campione.

⚠ **Confound da dichiarare, non da nascondere.** `stratify_by_size` considera
"rilevato" *qualsiasi* persona nel frame sopra soglia, non il bersaglio
specifico con matching IoU. Su VisDrone (pochi soggetti per frame) era
accettabile; su Okutama (fino a 9 attori simultanei) no: nei frame affollati
basta che YOLO veda un altro attore perché il target conti come "non evaso".
Probabile che il 46% del bucket dominante sia **sottostimato** e che il 96.8%
del bucket medio corrisponda a frame con meno attori (bersagli grandi = drone
più basso = meno gente inquadrata). **Da riportare come tendenza, non come
misura pulita.** `[APERTO]` la versione IoU-matched è un miglioramento
possibile e non costoso.

`[VERIFICATO]` **K ottimo diverso tra i due dataset — dettaglio onesto per la
tesi**, dai grafici `plot_k_selection.py` rigenerati su entrambi i dataset:

| Criterio | K ottimo VisDrone (640) | K ottimo Okutama (960) |
|---|---|---|
| F1 | 244 | ~55 |
| Media geometrica √(R1·R2) | 37 | ~10 |

`LOSS_TOP_K=20` resta nell'ordine di grandezza corretto per il criterio
primario su entrambi i dataset (10–37), ma su Okutama non è più "vicino al
valore esatto" come lo era specificamente su VisDrone — il nucleo di
confidenza reale (dove R2 crolla, visibile nel pannello in basso a destra dei
grafici) è più stretto. `[IPOTESI]` verosimilmente perché il canvas 960
concentra il segnale su un'area proporzionalmente diversa della griglia YOLO
rispetto a 640 — non verificato quantitativamente. Da riportare come
osservazione onesta, non come problema: la scelta K=20 resta valida su
entrambi i dataset, semplicemente con margine diverso.

### 7.9 Bug e debiti tecnici emersi in Fase 7

| # | Problema | Causa | Fix |
|---|---|---|---|
| 8 | `RuntimeError: size of tensor a (1280) must match b (640)` in `optimize_universal` | `IMG_SIZE` importato da `config.py` come costante **fissa a 640** e usato per costruire canvas, padding e telemetria — mai emerso perché VisDrone è sempre stato 640 | `img_size = img_t.shape[-1]` derivato dal tensore reale; sostituito in **tre** punti distinti (canvas, padding, blocco telemetria a step%20) |
| 9 | `F.interpolate ... output (H: -292, W: 11)` a step 140 | Terzo punto con lo stesso hardcode, nel blocco di telemetria visiva: bbox in spazio 960 clampata con `IMG_SIZE`=640 → rettangolo ad altezza negativa | stessa correzione, `img_size` dinamico |
| 10 | `cli.py`: `--img-size` definito **due volte** (default 960 e 1280) | due patch successive applicate senza rimuovere la precedente | rimossa la duplicata — verificato risolto |
| 11 | Coordinate Okutama scalate male in modo silenzioso | label in 3840×2160, frame in 1280×720: applicare la scala dell'immagine caricata a coordinate native dava bbox clampate ai bordi senza errore | scaling in un passaggio da 3840×2160 al canvas finale |
| 12 | Output multipli con nomi fissi si sovrascrivevano tra run diversi (`full_report.json`, `k_selection_raw.npz`, `training_metrics.json`); un file di Fase 6 VisDrone (`bootstrap_ci_report.json`, `vision_metrics.json`) è stato scambiato per un risultato Okutama durante l'analisi | Nessun campo dataset/config nei file, path sempre identici | Convenzione di naming auto-descrittiva (`<file>_<loader>[_<img_size>][_stride<N>].ext`), applicata a `cli.py`, `plot_k_selection.py`, `plot_CI_box.py`, `generate_before_after_images.py` — vedi Parte I |

**Osservazione metodologica non banale (bug #8):** i risultati VisDrone erano
implicitamente **legati alla risoluzione**. Il codice non era portabile ad altre
scale, e nessun test lo aveva rivelato perché esisteva un solo dataset a una
sola risoluzione. La migrazione ha reso la pipeline effettivamente
multi-risoluzione — un risultato collaterale della Fase 7, citabile.

**Debiti residui noti** `[APERTO]`:
- `_save_checkpoint` (salvataggio d'emergenza su `Ctrl+C`) scrive su
  `BEST_PATCH_FILE` fisso, non su `--patch-out`. Irrilevante per run
  completati; da correggere se si prevede di interrompere a mano.
- `tactical_preflight_check` è VisDrone-only (usa l'API interna
  `_parse_annotation`), quindi su Okutama viene saltato con un messaggio
  esplicito. Il pre-flight equivalente è fatto a mano (§7.3).
- `--eval-vision` (il comando che scrive il bridge `vision_metrics.json` per
  il simulatore) è ancora VisDrone-only, stesso pattern di loader fisso degli
  altri tool prima del fix. Da correggere **prima** di un'eventuale fase
  Layer 2/3 con dati Okutama (vedi Parte V).
- Sigla della metrica Layer 3 da disambiguare: `compute_clae` nel codice vs
  "CEAE" nel README — stessa metrica con nome incoerente, o due cose diverse?
  Verificare prima di scriverne in tesi.

### 7.10 Comandi per riprodurre tutta la Fase 7

```bash
# 1. Verifica loader
python test_okutama_loader.py --data data/okutama_train

# 2. Pre-flight (a 960, coerente con training/eval)
python tools/count_negative_candidates.py --data data/okutama_val --loader okutama --img-size 960

# 3. Backup patch VisDrone PRIMA di allenare
cp outputs/patches/care_kit_patch_universal.pt outputs/patches/patch_visdrone.pt

# 4. Training (~8000 step)
python cli.py --train-patch --loader okutama --train-dir data/okutama_train \
  --patch-out outputs/patches/patch_okutama.pt --img-size 960 --fresh

# 5. Report decorrelato (IL DATO DA CITARE)
python cli.py --eval-report --data data/okutama_val \
  --patch outputs/patches/patch_okutama.pt --loader okutama --img-size 960 --stride 27

# 6. Robustezza multi-soglia (ripetere con --conf-threshold 0.3 e 0.7)
python cli.py --eval-report --data data/okutama_val \
  --patch outputs/patches/patch_okutama.pt --loader okutama --img-size 960 \
  --stride 27 --conf-threshold 0.3

# 7. Stratificazione per scala
python tools/stratify_by_size.py --data data/okutama_val --loader okutama \
  --img-size 960 --patch outputs/patches/patch_okutama.pt

# 8. Grafico a candela (auto-nominato dal report di input)
REPORT_PATH=outputs/metrics/full_report_okutama_960_stride27.json python tools/plot_CI_box.py

# 9. Immagini prima/dopo per figure di tesi
python tools/generate_before_after_images.py --data data/okutama_val \
  --patch outputs/patches/patch_okutama.pt --loader okutama --img-size 960
```

---

# Parte III — Risultati consolidati

## Confronto side-by-side (conf=0.5, bootstrap appaiato 10.000 iter)

| Metrica | VisDrone (n=531, 640px) | Okutama (n=527, 960px) |
|---|---|---|
| Evasion rate PRE | 0.175 | **0.497** |
| Evasion rate POST | 0.425 | **0.777** |
| **Δ Evasion** | **+0.250** [+0.156, +0.346] | **+0.280** [+0.228, +0.332] |
| Sensitività R1 PRE → POST | 0.825 → 0.575 | 0.503 → 0.223 |
| Specificità R2 | 0.969 invariante | 0.960 invariante |
| √(R1·R2) Δ | −0.148 [−0.210, −0.090] | −0.232 [−0.278, −0.188] |
| F1 Δ | −0.168 [−0.242, −0.100] | −0.300 [−0.358, −0.244] |
| Danno collaterale | −0.063 a conf 0.5, **`[RITIRATO]`** (non replica a 0.3/0.7) | −0.053 a conf 0.5, **robusto** a 0.3 e 0.5 |
| Frame positivi nel valset | 15.1% | **57.0%** (a 960, coerente) |
| K ottimo (F1 / media geom.) | 244 / 37 | ~55 / ~10 |
| Significatività metriche primarie | p<0.0001 | p<0.0001 |

## ⚠ Valutazione onesta: cosa aggiunge davvero Okutama

Questa sezione esiste per evitare di sovravendere il secondo dataset. Va letta
prima di scrivere qualunque frase comparativa.

### Cosa aggiunge (reale, difendibile)

1. **Replica su dominio indipendente.** Il finding centrale — la patch degrada
   la detection in modo statisticamente significativo — si riproduce su un
   secondo dataset con condizioni di ripresa, altitudine e composizione
   completamente diverse. Δ +0.280 vs +0.250: **notevolmente vicini**. Per una
   tesi, un effetto replicato su due domini vale più di un numero più alto su
   uno solo.

2. **Risolve una domanda esplicitamente aperta.** In Fase 2 il bucket 150px+
   aveva 11 casi e le note stesse lo definivano "non conclusivo, pista
   esplorativa". Okutama porta il bucket rilevante a n=618 con un effetto
   ampio (96.8% vs 46.0%), e due misure indipendenti collocano il salto nella
   stessa regione di scala. **L'ipotesi di scala passa da speculazione a
   evidenza** (con il confound di §7.8 dichiarato).

3. **Un risultato positivo nuovo:** il danno collaterale, ritirato su
   VisDrone, è significativo e robusto su Okutama a 0.3 e 0.5. È l'unico punto
   in cui Okutama produce un risultato *nuovo* invece di una replica.

4. **Portabilità dimostrata della pipeline** (bug #8): il framework non era
   multi-risoluzione e ora lo è.

### Cosa NON aggiunge (e va detto)

1. **Non rompe il soffitto strutturale — lo conferma su un secondo dataset.**
   L'evasion assoluta POST è molto più alta (77.7% vs 42.5%), ma **il
   contributo dell'attacco è quasi identico** (+28pp vs +25pp). Ciò che cambia
   è la *baseline*, non la potenza dell'attacco. Formulazione corretta: il
   guadagno marginale di questo attacco appare **limitato a ~25–28 punti
   percentuali attraverso due dataset, due risoluzioni e due distribuzioni di
   scala diverse**. È una conclusione più forte di quella ottenibile con
   VisDrone da solo — ma è una conferma del limite, non il suo superamento.

2. **"4× più frame positivi" non significa "4× più potenza statistica".**
   14.210 frame sono video correlato; una volta decorrelati restano ~527
   campioni quasi-indipendenti, cioè **la stessa potenza statistica di
   VisDrone (531)**. Okutama non aumenta il campione effettivo: cambia il
   *dominio*, non la *numerosità*. Da dire esplicitamente prima che lo faccia
   notare un revisore.

3. **Introduce un problema nuovo: la baseline PRE è debole.** YOLOv8n manca
   metà delle persone su Okutama **senza alcun attacco** (49.7% di evasion
   naturale contro 17.5% su VisDrone). Attaccare un detector già compromesso
   è una condizione di test meno pulita.
   `[IPOTESI, non verificata]` domain gap del detector (COCO è addestrato
   prevalentemente su pose upright a livello del suolo, mentre molte azioni
   Okutama sono viste dall'alto e includono posture come sdraiato/seduto),
   oppure distorsione dovuta allo stretch 1280×720 → 960×960.
   **Non risolvibile entro lo scope**: richiederebbe fine-tuning di YOLO su
   dati aerei, cioè cambiare l'oggetto della tesi (si valuta la robustezza di
   un detector dato, non se ne costruisce uno migliore).
   `[APERTO]` verifica possibile a basso costo: stratificare la PRE-evasion per
   taglia bbox — se concentrata sui piccoli ⇒ effetto scala; se uniforme ⇒
   posa/dominio.

4. **Il regime "bersaglio molto grande" resta non testato su entrambi i
   dataset.** VisDrone: n=11. Okutama a 960: bucket vuoto. Qualunque
   affermazione su bersagli >150px è **extrapolazione**, non misura.

5. **Costo della limitazione hardware:** 960 invece di 1280 ha ridotto del 56%
   le bbox utilizzabili. Il dataset avrebbe potuto dare di più con più memoria.

### Formulazione consigliata per la tesi

> L'attacco produce un degrado statisticamente significativo e di entità
> comparabile su due dataset aerei indipendenti (+25 e +28 punti percentuali di
> evasion rate, entrambi p<0.0001). La consistenza del guadagno marginale
> attraverso domini, risoluzioni e distribuzioni di scala diverse supporta
> l'interpretazione del limite osservato come **strutturale** — legato alla
> dimensione in pixel del bersaglio in prospettiva aerea — piuttosto che come
> insufficienza dell'ottimizzazione.

**Da non scrivere:** "l'attacco è più efficace su Okutama". È attaccabile in
difesa: il delta è simile, cambia la baseline.

---

# Parte IV — Per la call col relatore

## La storia in 7 frasi

1. Un attacco adversarial patch universale contro YOLOv8n su VisDrone porta
   l'evasion rate dal 17.5% al 42.5% (Δ +0.250, p<0.0001) — effetto reale e
   statisticamente solido, ma non un collasso del sensore.
2. Sei configurazioni sistematicamente isolate (loss, aggregazione, accumulo,
   scheduler) convergono nella stessa banda stretta: il limite non è
   nell'ottimizzatore. La diagnosi diretta del gradiente (0.0007–0.005,
   clipping mai scattato) e la letteratura UAV-specific (nessun lavoro forte su
   VisDrone attacca i pedoni) indicano una causa strutturale: i pedoni in vista
   aerea occupano troppi pochi pixel.
3. La loss top-K raggiunge lo stesso massimo con metà del budget di calcolo, e
   una validazione a posteriori (`plot_k_selection.py`) mostra che K=20 era già
   dentro il nucleo di confidenza reale — non un'assunzione fortunata.
4. Due risultati sono stati **ritirati dopo verifica**: il danno collaterale su
   VisDrone (non robusto multi-soglia) e la ponderazione tattica della loss
   (nessun miglioramento, coerente con il 98.2% di annotazioni sotto soglia).
5. È emerso un limite strutturale del disegno sperimentale: **R2 non può
   rilevare l'attacco per costruzione**, perché la patch non viene mai disegnata
   nei frame senza bersaglio valido. R2 va riportato come caratterizzazione del
   detector, non come test dell'attacco.
6. La migrazione a Okutama-Action **replica** il risultato su un dominio
   indipendente (Δ +0.280, p<0.0001) e **risolve** la domanda aperta sulla scala
   (bucket 100–150px: 96.8% di evasion su n=618, contro 46.0% su n=19.921 nel
   bucket 60–100px).
7. Il contributo dell'attacco è quasi identico sui due dataset (+25 vs +28pp)
   pur con baseline e distribuzioni di scala molto diverse: **conferma** del
   soffitto strutturale su un secondo dominio, non suo superamento.

## Domande plausibili e risposte

**"Perché l'evasion rate non supera il 44% su VisDrone?"**
→ VisDrone è aereo, multi-scala, con pedoni di pochi pixel. Gradiente misurato
direttamente (debole), sei configurazioni convergenti, e la letteratura
UAV-specific conferma che nessun lavoro forte su VisDrone attacca i pedoni.

**"Come sapete che non è un bug del codice?"**
→ Il gradient clipping non è mai scattato (esclude esplosioni/errori di scala),
sei configurazioni metodologicamente diverse convergono nella stessa banda, e
il risultato si replica su un secondo dataset indipendente con la stessa
pipeline. Un bug isolato non produrrebbe quella coerenza.

**"Su Okutama l'evasion arriva al 78%: l'attacco è molto più forte?"**
→ No, e va detto chiaramente. Il *delta* è +28pp contro +25pp su VisDrone: quasi
identico. Ciò che cambia è la baseline — YOLO su Okutama parte già dal 49.7% di
evasion naturale. Il guadagno marginale dell'attacco appare limitato a ~25–28pp
su entrambi i domini.

**"Perché il detector fallisce già metà delle volte su Okutama senza attacco?"**
→ È un dato verificato (49.7%, CI [0.441, 0.553]). L'ipotesi più plausibile è un
domain gap di YOLOv8n/COCO sulle pose aeree overhead, ma **non l'ho verificata**
e non la presento come dimostrata. È un limite del detector sul dominio, non
dell'attacco, e risolverlo richiederebbe fine-tuning — fuori dallo scope di una
valutazione di robustezza su un detector dato.

**"Avete 14.210 frame di validation ma ne usate 527: perché?"**
→ Sono frame video a 30fps, quindi fortemente correlati: il bootstrap appaiato
li tratterebbe come indipendenti, restituendo CI artificialmente stretti
(pseudo-replicazione). Con stride 27 (~0.9s tra frame) le stime puntuali
restano invariate e i CI si allargano di ~5×, onestamente. Il campione
effettivo è quindi comparabile a VisDrone.

**"Perché 960×960 e non la risoluzione nativa o 1280?"**
→ Vincolo hardware verificato: a 1280 con EoT=16 il processo supera i 16GB di
memoria unificata e va in swap, rendendo il training non completabile. A 960
rientra (~12GB). 640 (parità esatta con VisDrone) è stato scartato con un
calcolo: avrebbe eliminato il 95% delle bbox valide, annullando il motivo della
migrazione. Il costo di 960 è dichiarato: −56% di bbox utilizzabili rispetto a
1280.

**"Avete provato ad aumentare ancora il training?"**
→ Sì: da 375 a 2500 update (6.6×), +2.5 punti con rendimenti già decrescenti.
Su Okutama, con 8000 step, l'ultimo miglioramento è avvenuto intorno allo step
5973: plateau prima della fine del budget.

**"Il framework serve solo per questo attacco?"**
→ No — la Vision alimenta un simulatore multi-agente (Fusion bayesiano
Vision+OSINT+Behavioral, soglie IHL) tramite un bridge JSON, visibile con
`--run-sim`. Layer non ancora aggiornato con i dati Okutama in questa sessione
(vedi Parte V).

## Punti di forza da rivendicare

- **Metodo sistematico**: configurazioni isolate una variabile alla volta, non
  tuning casuale fino al numero migliore.
- **Diagnosi quantitativa, non intuitiva**: norma del gradiente misurata, non
  dedotta.
- **Due risultati ritirati dopo verifica di robustezza.** È il punto di forza
  meno ovvio e più solido: dimostra che i risultati positivi riportati hanno
  superato controlli che altri non hanno superato.
- **Un limite strutturale del proprio disegno sperimentale scoperto e
  documentato** (R2 tautologico), invece di riportare "R2 invariato = attacco
  pulito" come farebbe una lettura superficiale.
- **Replica su dominio indipendente**, con confronto side-by-side e senza
  accorpamento dei dataset.
- **Correzione della pseudo-replicazione** su dati video, prima che diventasse
  un risultato pubblicato.
- **Framework multi-dominio**: la Vision è un layer di un sistema più ampio
  (Fusion bayesiano, OSINT poisoning, soglie IHL) che la letteratura sulle
  adversarial patch generalmente non modella.

---

# Parte V — Aperto / da fare

Solo elementi **effettivamente aperti**. Tutto ciò che è completato sta nel
diario (Parte II).

## Fase successiva, non ancora iniziata: Layer 2/3 con dati Okutama

Il simulatore (`--run-sim`), la Fusion Bayesiana e la metrica CEAE/CLAE sono
codice pre-esistente, **non toccato in questa sessione** (che era interamente
Layer 1/Vision). Per portarli avanti con i dati Okutama:

1. Disambiguare CEAE vs CLAE (nome incoerente tra codice e README, vedi §7.9).
2. `--eval-vision` (bridge `vision_metrics.json`) è ancora VisDrone-only —
   stesso fix `--loader`/`--img-size` già applicato altrove.
3. Decidere se questa è materiale per lo stesso capitolo di tesi o per un
   capitolo separato — la sequenza dei lavori in questa sessione non lo
   presupponeva.

`[APERTO]` Da trattare come fase di lavoro distinta, per non destabilizzare la
chiusura del capitolo Vision appena raggiunta.

## Da chiedere/confermare col relatore

- `[APERTO]` **Grafico "a candela" per le metriche bootstrap.** Il relatore lo
  aveva menzionato durante la fase VisDrone (punto con baffi = stima puntuale +
  CI95%). Implementato in `tools/plot_CI_box.py` (forest plot pubblicazione-
  ready, pannello (a) livelli assoluti + pannello (b) delta appaiato). Da
  confermare col relatore se il formato è quello inteso.
- `[APERTO]` Verificare che il confronto side-by-side nella forma della Parte
  III sia quello atteso (colonne, versione filtrata/non filtrata per
  dimensione).

## Verifiche a basso costo, non bloccanti

- `[APERTO]` Stratificare la **PRE**-evasion di Okutama per taglia bbox: test
  diretto dell'ipotesi domain-gap (uniforme) contro effetto scala (concentrato
  sui piccoli). Chiarirebbe §7.8 e la nota su "cosa non aggiunge" punto 3.
- `[APERTO]` Versione **IoU-matched** di `stratify_by_size.py`: eliminerebbe il
  confound "altri attori nel frame" che su Okutama (fino a 9 soggetti) è
  significativo.
- `[APERTO]` R2 bootstrap con i 6.115 frame ambigui come classe negativa
  estesa su Okutama, per parità con il trattamento VisDrone (attualmente non
  ancora eseguito esplicitamente con questo n).
- `[APERTO]` Esito del retest di ponderazione tattica su Okutama (menzionato
  come "in corso" — se concluso, va aggiunto in Fase 6 con lo stesso rigore
  degli altri risultati, pre-flight + verifica di robustezza).

## Lavoro futuro (non per questa tesi)

- Loss che copra l'intera impronta geometrica del bersaglio (K≈244 su
  VisDrone, ~55 su Okutama) invece del solo nucleo di confidenza — nuovo ciclo
  sperimentale, esplicitamente rinviato.
- Loss su feature intermedie del backbone invece dell'output finale (tecnica
  del paper 2023 su aerial imagery) — richiede hook sui layer intermedi.
- Dataset custom image-specific (`tools/annotate_mioDS.py`, Wu et al. 2020):
  converge più in fretta perché non deve generalizzare su scene eterogenee.
  Non eseguito (richiede raccolta foto manuale); archiviato ma non
  cancellato, resta citabile come alternativa di design.
- Correggere `_save_checkpoint` per rispettare `--patch-out`.
- Correggere `--eval-vision` per supportare `--loader`/`--img-size` (necessario
  per la fase Layer 2/3 con Okutama).

---

# Letteratura citata

| Fonte | Uso nel progetto |
|---|---|
| Barekatain et al. 2017 | Okutama-Action: dataset aereo, altitudine 10–45m (Fase 7) |
| Sodhro et al. 2025 | Baseline YOLOv8 outdoor 99.1% confidence |
| Carlini & Wagner 2017 | Margine di robustezza nella loss |
| Wu et al. 2020 | Domain-Specific Attacks > Universal Attacks (Piano B) |
| Athalye et al. 2017 | EoT |
| Thys et al. 2019 | TV Loss, aggregazione max/top-K |
| Brown et al. 2017 | Adversarial Patch, convergenza e rendimenti marginali |
| Arkin 2009 | Principio di Distinzione IHL, scenario Urban Clutter |
| Liu et al. 2019 | DPatch — loss congiunta box+classe (riserva) |
| Huang et al. 2020 | Universal Physical Camouflage — design region-aware |
| Hu et al. 2021 | Naturalistic Patch — GAN latent space |
| Shrestha, Pathak, Viegas 2023 | Patch UAV-specific su VisDrone (Car, 80% ASR) |
| LFRAP 2025 | Conferma: pedoni scartati per dimensione da vista aerea |
| Everingham et al. 2010 | Standard IoU=0.5 (PASCAL VOC), da cui ci discostiamo |
| Yu et al. 2020 | "Scale Match for Tiny Person Detection" — IoU permissivo |
| Shao et al. 2018 | CrowdHuman — convenzione "ignore region" |
| Efron & Tibshirani 1993 | Bootstrap appaiato |

DOI/URL completi in `config.py`.
