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
- ✅ = chiuso, con la fonte della chiusura indicata.

**Scope**: Layer 1 (Vision) è chiuso e validato su due dataset. Layer 2/3
(Simulation, Fusion bayesiana OSINT, CEAE) è stato eseguito ma i suoi numeri
**non sono ancora validi** — vedi Fase 8 e `handoff_simulatore_layer23.md`.

---

# Parte I — Il sistema

## Architettura: tre layer disaccoppiati

Collegati da file JSON, non da chiamate dirette:

1. **Vision** (`patch_optimizer.py`) — training di una Universal Adversarial
   Patch contro YOLOv8n, EoT completo (16 trasformazioni), loss asintotica.
2. **Simulation** (`simulator.py`, `entities.py`, `detection.py`) — ambiente
   multi-agente che legge il degrado empirico da `vision_metrics.json`.
3. **Fusion/Decision** (`fusion_decision.py`) — sensor fusion bayesiano
   (Vision/OSINT/Behavioral) con soglie di ingaggio IHL, metrica **CEAE**
   (Cost-Effective Adversarial Engagement). Nota: nel codice la funzione si
   chiama `compute_clae` per ragioni storiche — **il nome corretto da usare in
   tesi è CEAE**, scelto per non confondersi con metriche esistenti omonime.

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
  (Fase 3).
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
  visivo è solo data-driven, dal drop empirico misurato.

## Glossario

- **Step raw**: un ciclo del training loop (immagine → EoT → forward →
  backward). `PATCH_STEPS` conta questi, non gli aggiornamenti dei pixel.
- **Gradient Accumulation**: quanti step raw si sommano prima di un update
  reale. `update_reali = step_raw / accumulation_steps`. Con N=4 servono 4
  immagini prima che i pixel della patch cambino davvero una volta — simula un
  batch più grande senza il costo di memoria di caricarlo tutto insieme, al
  prezzo di meno aggiornamenti reali a parità di immagini viste.
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
- **Copertura tattica**: frazione delle annotazioni già valide (≥60px) che sono
  anche ≥80px. ⚠ **Non confrontabile tra dataset a risoluzioni diverse** —
  vedi §7.8.
- **Norma del gradiente / gradient clipping**: la norma è un singolo numero che
  misura quanto è "grande" il vettore gradiente nel complesso — quanto è forte
  il segnale di aggiornamento in un dato istante. Il clipping è una rete di
  sicurezza: se la norma supera 1.0, viene ridimensionata per evitare update
  troppo bruschi. Non è mai scattata nei nostri esperimenti — segno che il
  problema non è mai stato un gradiente troppo grande, ma uno troppo debole.

## Glossario dei tool (`tools/`)

Aggiunto perché con due dataset e più run il numero di script cresce e i nomi
da soli non bastano a ricordare cosa fa cosa.

| Tool | Cosa fa | Quando usarlo |
|---|---|---|
| `count_negative_candidates.py` | Conta frame positivi/negativi/ambigui | Prima di allenare, verifica campione |
| `stratify_by_size.py` | Evasion rate per fascia di altezza bbox, con una patch già addestrata | Diagnosi soffitto strutturale/scala |
| `plot_k_selection.py` | Curve F1/√(R1·R2)/R1/R2 al variare di K (nessuna patch, comportamento naturale) | Scelta/verifica di `LOSS_TOP_K` |
| `bootstrap_ci_report.py` | CI bootstrap PRE/POST, versione precedente (solo F1+gmean, no R1/R2 separate) | Superato da `--eval-report`, tenuto per lo storico di Fase 6 |
| `plot_CI_box.py` | Grafico a candela / forest plot (punto + baffi CI95%) da `full_report_*.json` | Figura per la tesi, dopo `--eval-report` |
| `plot_training_curves.py` | Curva Loss/TV Loss durante un training, da `training_metrics_*.json` | Dopo `--train-patch`, se serve il grafico |
| `plot_runs_comparison.py` | Barre di confronto tra le 6 configurazioni storiche (lista `RUNS` scritta a mano nello script) | Solo per il confronto Fase 1-2, statico, non dipende da file generati |
| `generate_before_after_images.py` | Immagini affiancate PRE/POST con box disegnate, crop leggibile | Figure per slide/tesi, prova visiva oltre alle tabelle |
| `annotate_mioDS.py` | "Piano B": auto-annotazione foto raw in formato VisDrone (Wu et al. 2020) | Non usato — solo citato come alternativa di design in "Lavoro futuro" |

## Convenzione di naming degli output (adottata in Fase 7, tardi ma adottata)

Problema che ha causato confusione reale in sessione: molti tool scrivevano su
path fissi (`full_report.json`, `k_selection_raw.npz`, `training_metrics.json`)
senza indicare dataset/config nel nome — un run successivo sovrascriveva
silenziosamente quello precedente, e file di run diversi (es. Fase 6 VisDrone
vs Okutama) sono stati scambiati per errore durante l'analisi.

**Convenzione adottata**: `<nome_base>_<loader>[_<img_size>][_stride<N>].ext`,
stessa cartella `outputs/metrics/` (nessuna sottocartella, minima
ristrutturazione). Esempio: `full_report_okutama_960_stride27.json`. I file
storici con nome generico sono stati spostati in
`outputs/metrics/archive_fase6_visdrone/`.

**Eccezione deliberata**: `vision_metrics.json` mantiene il nome fisso. Non è
un artefatto di analisi ma il **bridge di pipeline** letto dal simulatore —
c'è sempre un solo "risultato attivo corrente" per costruzione, quindi non
esistono run multipli da distinguere. Distinzione architetturale voluta:
`full_report_*.json` è un artefatto di analisi per la tesi, `vision_metrics.json`
è un database di pipeline. Cicli di vita diversi, non vanno confusi.

**Perdita nota, irreversibile**: `training_metrics.json` (path fisso prima del
fix) è stato sovrascritto più volte — la curva di loss del run VisDrone finale
(Fase 2, config #5) non è più rigenerabile. Non impatta i risultati numerici
(già consolidati come tabelle), solo un grafico di supporto. Da ora ogni
`--train-patch` salva una copia con suffisso.

---

# Parte II — Diario di viaggio

## Fase 0 — Far funzionare la pipeline su Apple Silicon

Hardware: MacBook Air M4, 16GB di memoria **unificata** (CPU e GPU la
condividono — vincolo che tornerà decisivo in Fase 7).

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
sua scoperta usavano uno scheduler allungato**, quindi la cronologia sotto
indica esplicitamente quali run avevano lo scheduler corretto e quali no.

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
**Conseguenza: nessun nuovo training.** K=244 resta annotato come lavoro futuro
(una loss che copra l'intera impronta geometrica), non come azione.

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
della patch — l'ha reso più preciso su una quantità che per costruzione non può
muoversi.** Il vecchio risultato "R2=1.0 invariato" con n=9 non era la
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

**Nota per un eventuale retest su Okutama** (discusso, non concluso): il fattore
limitante qui identificato — troppo pochi bersagli grandi nel trainset — è
quantitativamente diverso su Okutama (il bucket 100–150px ha 598 casi nel solo
valset, contro 394 ≥80px in tutto il trainset VisDrone). Se un retest viene
completato, il risultato va aggiunto qui con lo stesso rigore (pre-flight +
verifica di robustezza) prima di scriverlo in tesi.

**VisDrone si chiude qui.** Pipeline validata, numeri definitivi, non più
toccata.

---

## Fase 7 — Migrazione a Okutama-Action

Richiesta del relatore: testare un dataset con pedoni ripresi più da vicino
**separatamente**, e confrontare. Non accorpare, non sostituire. Se
statisticamente non migliora, è comunque un risultato valido.

Candidato scelto: **Okutama-Action** (Barekatain et al., CVPR-W 2017;
altitudine 10–45m, coerente con `DRONE_ALTITUDE_M=10`; CC BY-NC-SA 3.0).
Dataset SAR (HERIDAL, SARD, LADD) scartati: stesso problema di oggetti piccoli.

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
float32 il processo raggiunge **~17 GB residenti su 16 GB totali** → swap
continuo (contatore `swapouts` in crescita costante, processo in stato `stuck`,
throughput crollato: fermo allo step 40 dopo un'ora). Non "lento": non
completabile.

`[VERIFICATO]` **960×960 rientra nel budget:** ~12 GB residenti, delta
`swapouts` = 0, processo `running`, 8000 step completati.

**640×640 (parità esatta con VisDrone) scartato con un calcolo, non a
sensazione:** a 640 ogni altezza bbox si dimezza rispetto a 1280, quindi una
bbox resta ≥60px solo se era ≥120px a 1280. Dalla distribuzione misurata a 1280
questo cancella l'intero bucket 60–100px (87.9% del totale): le bbox valide
crollerebbero da 46.293 a ~2.400, **−95%**, riportando il problema che la
migrazione doveva risolvere.

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
in precedenza a 1280 con la patch **VisDrone** dava 60–100px: 52.2% (n=40683),
100–150px: 48.6% (n=5294), 150px+: **97.2%** (n=316). Convertendo le scale
(960 → 1280 = ×1.333), il bucket 100–150@960 corrisponde a 133–200@1280 — cioè
proprio la regione dove la misura a 1280 mostrava il salto al 97%. **Due misure
indipendenti, patch diverse e risoluzioni diverse, individuano il salto nella
stessa regione di scala.**

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
possibile e non costoso — **è lavoro Vision, da fare prima di scrivere i
capitoli Risultati/Discussione**, perché rafforza il finding più forte che
Okutama porta.

`[VERIFICATO]` **Distribuzione di scala: il guadagno reale è nei numeri
assoluti, non nella forma della distribuzione.** Bbox valide (≥60px) nel
valset: VisDrone 381 su 531 frame (**0.72/frame**); Okutama 20.539 su 14.210
frame (**1.45/frame**) — circa **2× per frame**, 54× in assoluto. Ma la *forma*
non è migliore: su Okutama il 97.0% delle bbox valide sta nella banda 60–100px,
contro l'82.4% di VisDrone.

⚠ **Le coperture tattiche (≥80px) NON sono confrontabili tra i due dataset**:
15.2% su Okutama a 960 contro 32.0% su VisDrone a 640. Il rapporto
(≥80px)/(≥60px) dipende dalla forma della distribuzione attorno alle soglie, e
soglie fisse in pixel campionano parti diverse di distribuzioni scalate in modo
diverso. Su Okutama a 960 la soglia di 60px cade quasi sulla **moda** della
distribuzione, rendendo ogni rapporto derivato ipersensibile a piccoli
spostamenti di soglia. Da riportare come **caveat metodologico**: in tesi
riportare la copertura tattica per dataset **senza confrontarla**, e usare i
conteggi assoluti (0.72 vs 1.45 bbox valide per frame) quando serve argomentare
che Okutama è il dataset migliore. Qualunque lettura del tipo "Okutama ha
bersagli relativamente più piccoli" sarebbe un artefatto della scala, non una
misura.

`[VERIFICATO]` **K ottimo diverso tra i due dataset — dettaglio onesto per la
tesi**, dai grafici `plot_k_selection.py` rigenerati su entrambi i dataset:

| Criterio | K ottimo VisDrone (640) | K ottimo Okutama (960) |
|---|---|---|
| F1 | 244 | ~55 |
| Media geometrica √(R1·R2) | 37 | ~10 |

`LOSS_TOP_K=20` resta nell'ordine di grandezza corretto per il criterio
primario su entrambi i dataset (10–37), ma su Okutama non è più "vicino al
valore esatto" come lo era specificamente su VisDrone — il nucleo di confidenza
reale (dove R2 crolla, visibile nel pannello in basso a destra dei grafici) è
più stretto. `[IPOTESI]` verosimilmente perché il canvas 960 concentra il
segnale su un'area proporzionalmente diversa della griglia YOLO rispetto a 640
— non verificato quantitativamente. Da riportare come osservazione onesta, non
come problema: la scelta K=20 resta valida su entrambi i dataset, semplicemente
con margine diverso.

`[VERIFICATO]` **Specificità dell'attacco al bersaglio patchato — evidenza
visiva.** Nelle figure prima/dopo generate su Okutama, in un frame con due
persone (una con patch, una senza, sotto soglia 60px) il target patchato evade
completamente mentre **la seconda persona resta rilevata con confidenza
invariata**. Osservazione qualitativa su singolo frame, non una misura
aggregata, ma coerente in direzione con il segno negativo del danno collaterale
(§7.7): l'attacco degrada la detection del bersaglio su cui è applicato, non il
comportamento del detector sull'intera scena. Utile come figura a supporto
della discussione sul danno collaterale.

### 7.9 Bug e debiti tecnici emersi in Fase 7

| # | Problema | Causa | Fix |
|---|---|---|---|
| 8 | `RuntimeError: size of tensor a (1280) must match b (640)` in `optimize_universal` | `IMG_SIZE` importato da `config.py` come costante **fissa a 640** e usato per costruire canvas, padding e telemetria — mai emerso perché VisDrone è sempre stato 640 | `img_size = img_t.shape[-1]` derivato dal tensore reale; sostituito in **tre** punti distinti (canvas, padding, blocco telemetria a step%20) |
| 9 | `F.interpolate ... output (H: -292, W: 11)` a step 140 | Terzo punto con lo stesso hardcode, nel blocco di telemetria visiva: bbox in spazio 960 clampata con `IMG_SIZE`=640 → rettangolo ad altezza negativa | stessa correzione, `img_size` dinamico |
| 10 | `cli.py`: `--img-size` definito **due volte** (default 960 e 1280) | due patch successive applicate senza rimuovere la precedente | rimossa la duplicata — verificato risolto |
| 11 | Coordinate Okutama scalate male in modo silenzioso | label in 3840×2160, frame in 1280×720: applicare la scala dell'immagine caricata a coordinate native dava bbox clampate ai bordi senza errore | scaling in un passaggio da 3840×2160 al canvas finale |
| 12 | Output multipli con nomi fissi si sovrascrivevano tra run diversi (`full_report.json`, `k_selection_raw.npz`, `training_metrics.json`); un file di Fase 6 VisDrone (`bootstrap_ci_report.json`, `vision_metrics.json`) è stato scambiato per un risultato Okutama durante l'analisi | Nessun campo dataset/config nei file, path sempre identici | Convenzione di naming auto-descrittiva (vedi Parte I), applicata a `cli.py`, `plot_k_selection.py`, `plot_CI_box.py`, `generate_before_after_images.py` |

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

# 8. Forest plot / grafico a candela (auto-nominato dal report di input)
REPORT_PATH=outputs/metrics/full_report_okutama_960_stride27.json python tools/plot_CI_box.py

# 9. Immagini prima/dopo per figure di tesi
#    ⚠ su Okutama serve --max-samples alto: i frame sono ordinati per video,
#    i primi 200 appartengono a un solo scenario e potrebbero non contenere
#    il pattern cercato (persona singola, confidenza >= soglia).
python tools/generate_before_after_images.py --data data/okutama_val \
  --patch outputs/patches/patch_okutama.pt --loader okutama --img-size 960 \
  --max-samples 14210
```

---

## Fase 8 — Layer 2/3: stato accertato, non ancora risolto

`[VERIFICATO]` Eseguito `--eval-vision` + `--run-sim` su Okutama (960, valset
completo senza stride). Vision: F1=0.354, Precision=0.906, Recall=0.220
(TP=1781, FN=6314 su 8095 frame positivi), evasion 78.0%, evasion ≥80px 82.5%,
copertura tattica 15.2%. L'evasion completo **coincide alla cifra con il POST di
§7.6** (0.7800) — buon controllo di consistenza tra due comandi indipendenti.

⚠ **Le tabelle CEAE prodotte da `--run-sim` NON sono utilizzabili in tesi così
come sono.** Quattro difetti accertati per lettura del codice (dettaglio,
diagnosi e piano di intervento in `handoff_simulatore_layer23.md`):

1. **D1 — la detection non è un test statistico.** `detection.py` usa
   `random.uniform(0, max(F1, 0.05))` con soglia fissa: il collasso a TP=0 è
   deterministico per design, e F1 (media armonica aggregata di precision e
   recall) non ha interpretazione probabilistica per singolo frame. La quantità
   corretta per "il target viene visto?" è **R1**.
2. **D2 — la baseline non legge mai il PRE reale.** È ancorata alla costante
   `0.710` (Sodhro et al. 2025); il JSON viene letto **solo** negli scenari
   sotto attacco. Le tabelle confrontano quindi un riferimento di letteratura
   generico con il dataset sotto attacco, non PRE vs POST — il delta risulta
   inquinato dal domain gap, esattamente l'effetto che nel capitolo Vision
   abbiamo isolato con cura.
3. **D3 — R2 non entra mai nel simulatore, in nessuno scenario.** I civili non
   ricevono mai `care_kit_active`, quindi cadono sempre nel ramo "sensore in
   salute" (`uniform(0.70, 0.95)`). Metà del lavoro metrico fatto col relatore
   non raggiunge Layer 2/3. Coerente con l'osservazione empirica: gli FP restano
   quasi costanti tra scenari (3279 in Esp. 1; 144 in Esp. 2 baseline e patch).
4. **D4 — doppio conteggio dell'effetto distanza.** `dist_scale` è applicato
   alla confidenza, ma R1 è già una media empirica su frame a distanze
   eterogenee.

**Ipotesi non verificata, da NON scrivere come fatto:** la discrepanza tra
Esp. 1 (collasso totale) ed Esp. 2 (impatto quasi nullo) potrebbe dipendere dal
residuo vision in `fusion_decision.py` (`v = confidence * 0.5` quando
`detected=False`) compensato da OSINT+Behavioral al 55%. **Verifica rinunciata
deliberatamente** (vedi handoff dedicato): in tesi riportare solo la
formulazione descrittiva "il sistema fuso mostra un degrado marcatamente
inferiore rispetto al canale visivo isolato".

`[APERTO]` Intervento pianificato e scoped (F1–F4 nell'handoff dedicato). Fino
ad allora i numeri di `--run-sim` restano proof-of-concept qualitativo.

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
| Frame positivi nel valset | 15.1% | **57.0%** |
| Bbox valide (≥60px) per frame | 0.72 | **1.45** |
| K ottimo (F1 / media geom.) | 244 / 37 | ~55 / ~10 |
| Significatività metriche primarie | p<0.0001 | p<0.0001 |

⚠ Non inserire in questa tabella la copertura tattica (≥80px): non è
confrontabile tra dataset a risoluzioni diverse (§7.8).

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
   esplorativa". Okutama porta il bucket rilevante a n=618 con un effetto ampio
   (96.8% vs 46.0%), e due misure indipendenti collocano il salto nella stessa
   regione di scala. **L'ipotesi di scala passa da speculazione a evidenza**
   (con il confound di §7.8 dichiarato).

3. **Un risultato positivo nuovo:** il danno collaterale, ritirato su VisDrone,
   è significativo e robusto su Okutama a 0.3 e 0.5. È l'unico punto in cui
   Okutama produce un risultato *nuovo* invece di una replica.

4. **Materiale utilizzabile: ~2× bersagli validi per fotogramma** (1.45 vs
   0.72), e 4× frame positivi (57.0% vs 15.1%). Questo è il dato solido da usare
   quando serve argomentare la scelta del dataset — non le coperture tattiche.

5. **Portabilità dimostrata della pipeline** (bug #8): il framework non era
   multi-risoluzione e ora lo è.

### Cosa NON aggiunge (e va detto)

1. **Non rompe il soffitto strutturale — lo conferma su un secondo dataset.**
   L'evasion assoluta POST è molto più alta (77.7% vs 42.5%), ma **il contributo
   dell'attacco è quasi identico** (+28pp vs +25pp). Ciò che cambia è la
   *baseline*, non la potenza dell'attacco. Formulazione corretta: il guadagno
   marginale di questo attacco appare **limitato a ~25–28 punti percentuali
   attraverso due dataset, due risoluzioni e due distribuzioni di scala
   diverse**. È una conclusione più forte di quella ottenibile con VisDrone da
   solo — ma è una conferma del limite, non il suo superamento.

2. **"4× più frame positivi" non significa "4× più potenza statistica".**
   14.210 frame sono video correlato; una volta decorrelati restano ~527
   campioni quasi-indipendenti, cioè **la stessa potenza statistica di VisDrone
   (531)**. Okutama non aumenta il campione effettivo: cambia il *dominio*, non
   la *numerosità*. Da dire esplicitamente prima che lo faccia notare un
   revisore.

3. **La forma della distribuzione di scala non è migliore.** Il 97.0% delle
   bbox valide su Okutama sta nella banda 60–100px (82.4% su VisDrone): il
   guadagno è in quantità assoluta, non in "bersagli più grandi in proporzione".

4. **Introduce un problema nuovo: la baseline PRE è debole.** YOLOv8n manca
   metà delle persone su Okutama **senza alcun attacco** (49.7% di evasion
   naturale contro 17.5% su VisDrone). Attaccare un detector già compromesso è
   una condizione di test meno pulita.
   `[IPOTESI, non verificata]` domain gap del detector (COCO è addestrato
   prevalentemente su pose upright a livello del suolo, mentre molte azioni
   Okutama sono viste dall'alto e includono posture come sdraiato/seduto),
   oppure distorsione dovuta allo stretch 1280×720 → 960×960.
   **Non risolvibile entro lo scope**: richiederebbe fine-tuning di YOLO su
   dati aerei, cioè cambiare l'oggetto della tesi (si valuta la robustezza di un
   detector dato, non se ne costruisce uno migliore).
   `[APERTO]` verifica possibile a basso costo: stratificare la PRE-evasion per
   taglia bbox — se concentrata sui piccoli ⇒ effetto scala; se uniforme ⇒
   posa/dominio.

5. **Il regime "bersaglio molto grande" resta non testato su entrambi i
   dataset.** VisDrone: n=11. Okutama a 960: bucket vuoto. Qualunque
   affermazione su bersagli >150px è **extrapolazione**, non misura.

6. **Costo della limitazione hardware:** 960 invece di 1280 ha ridotto del 56%
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

## Feedback ricevuto (mail, luglio) — registrato

> "è tutto ben fatto e sì, i grafici sono esattamente quelli che mi aspettavo.
> Nel nostro caso il delta è sicuramente più interessante in quanto le
> differenze sono appaiate, ma far vedere R1 e R2 (e i loro CI) assieme ai
> delta va benissimo. Procedi pure."

Tre conseguenze operative:
1. ✅ Il formato del forest plot è **approvato**. `[APERTO]` chiuso.
2. **Il delta ha priorità narrativa**: in tesi dare rilievo al pannello del
   delta appaiato, o citarlo per primo nel testo che accompagna la figura.
3. `[APERTO]` Il relatore nomina **R2**, che nel forest plot attuale non
   compare (le metriche sono Evasion, R1, √(R1·R2), F1). Vedi Parte V.

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
sei configurazioni metodologicamente diverse convergono nella stessa banda, e il
risultato si replica su un secondo dataset indipendente con la stessa pipeline.
Un bug isolato non produrrebbe quella coerenza.

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
(pseudo-replicazione). Con stride 27 (~0.9s tra frame) le stime puntuali restano
invariate e i CI si allargano di ~5×, onestamente. Il campione effettivo è
quindi comparabile a VisDrone.

**"Perché 960×960 e non la risoluzione nativa o 1280?"**
→ Vincolo hardware verificato: a 1280 con EoT=16 il processo supera i 16GB di
memoria unificata e va in swap, rendendo il training non completabile. A 960
rientra (~12GB). 640 (parità esatta con VisDrone) è stato scartato con un
calcolo: avrebbe eliminato il 95% delle bbox valide, annullando il motivo della
migrazione. Il costo di 960 è dichiarato: −56% di bbox utilizzabili rispetto a
1280.

**"La copertura tattica su Okutama è più bassa: il dataset è peggiore?"**
→ No, quel confronto non è valido. La copertura è un rapporto tra soglie fisse
in pixel su distribuzioni scalate diversamente (640 vs 960), e su Okutama la
soglia di 60px cade quasi sulla moda della distribuzione. Il dato corretto per
confrontare i dataset sono i conteggi assoluti: 1.45 bbox valide per frame su
Okutama contro 0.72 su VisDrone.

**"Avete provato ad aumentare ancora il training?"**
→ Sì: da 375 a 2500 update (6.6×), +2.5 punti con rendimenti già decrescenti.
Su Okutama, con 8000 step, l'ultimo miglioramento è avvenuto intorno allo step
5973: plateau prima della fine del budget.

**"Il framework serve solo per questo attacco?"**
→ No — la Vision alimenta un simulatore multi-agente (Fusion bayesiano
Vision+OSINT+Behavioral, soglie IHL) tramite un bridge JSON, visibile con
`--run-sim`. ⚠ Quel layer ha però difetti accertati non ancora corretti (Fase
8): i suoi numeri non sono ancora presentabili come risultati.

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

## Chiusi di recente

- ✅ **Grafico a candela / forest plot confermato dal relatore** (mail, luglio):
  "i grafici sono esattamente quelli che mi aspettavo". Vedi Parte IV.
- ✅ **Pre-flight Okutama ricalcolato a 960** (57.0%, non più 87.6% misurato a
  1280): `count_negative_candidates.py` ha ora `--img-size`. Vedi §7.3.
- ✅ **Nome della metrica Layer 3**: si usa **CEAE** (Cost-Effective
  Adversarial Engagement), non CLAE — scelto per non confondersi con metriche
  esistenti omonime. Il codice mantiene `compute_clae` per ragioni storiche.

## Da chiedere/confermare col relatore

- `[APERTO]` **Includere R2 nel forest plot?** Il relatore ha nominato
  esplicitamente R1 **e R2**; il plot attuale mostra Evasion, R1, √(R1·R2), F1.
  Pro: essendo R2 invariante per costruzione, una riga piatta accanto a quattro
  che crollano renderebbe **visivo** il finding di Fase 4 — il risultato
  metodologico più originale del lavoro. Contro: rischia di sembrare un errore
  di plotting. Propensione: aggiungerla, con didascalia che spiega l'invarianza.
- `[APERTO]` **Formato figure multi-pannello**: le linee guida LaTeX richiedono
  un file immagine separato per pannello (`\subfloat` + pacchetto `subfig`), ma i
  nostri script producono PNG compositi (4 pannelli nel K-plot, 2 nel forest
  plot, 2 nel before/after). Chiedere se è accettabile trattare il composito
  come figura a pannello singolo con didascalia descrittiva, prima di rifare
  tutte le figure.
- `[APERTO]` Verificare che il confronto side-by-side nella forma di Parte III
  sia quello atteso (colonne, versione filtrata/non filtrata per dimensione).

## Verifiche a basso costo, non bloccanti

- `[APERTO]` **`stratify_by_size.py` con IoU-matching** (soglia 0.3, come nelle
  metriche): eliminerebbe il confound "altri attori nel frame" che su Okutama
  (fino a 9 soggetti) è significativo. **Priorità alta**: è lavoro Vision, non
  simulatore, e rafforza il finding più forte del lavoro. Da fare prima di
  scrivere i capitoli Risultati/Discussione.
- `[APERTO]` Stratificare la **PRE**-evasion di Okutama per taglia bbox: test
  diretto dell'ipotesi domain-gap (uniforme) contro effetto scala (concentrato
  sui piccoli).
- `[APERTO]` R2 bootstrap con i 6.115 frame ambigui come classe negativa estesa
  su Okutama, per parità con il trattamento VisDrone.
- `[APERTO]` **Licenza VisDrone** per l'inclusione di frame in tesi: mai
  verificata. Okutama è CC BY-NC-SA 3.0 (uso accademico non commerciale con
  attribuzione — citare Barekatain et al. 2017 **nella didascalia**, non solo in
  bibliografia). Le linee guida del relatore vietano immagini protette da
  copyright.
- `[APERTO]` Salvare i grafici matplotlib anche in **PDF** (vettoriale) oltre al
  PNG: le linee guida lo raccomandano esplicitamente. Una riga per script.
- `[APERTO]` Esito del retest di ponderazione tattica su Okutama, se completato.

## Fase successiva: Layer 2/3

`[APERTO]` Ristrutturazione del simulatore secondo `handoff_simulatore_layer23.md`
(interventi F1–F4). Da affrontare come **sessione dedicata**, quando si arriva a
scrivere §3.6 della tesi. Non blocca il resto della stesura.

## Lavoro futuro (non per questa tesi)

- Loss che copra l'intera impronta geometrica del bersaglio (K≈244 su VisDrone,
  ~55 su Okutama) invece del solo nucleo di confidenza — nuovo ciclo
  sperimentale, esplicitamente rinviato.
- Loss su feature intermedie del backbone invece dell'output finale (tecnica del
  paper 2023 su aerial imagery) — richiede hook sui layer intermedi.
- Dataset custom image-specific (`tools/annotate_mioDS.py`, Wu et al. 2020):
  converge più in fretta perché non deve generalizzare su scene eterogenee. Non
  eseguito (richiede raccolta foto manuale); **archiviato ma non cancellato** —
  resta citabile come alternativa di design, ed è format-specifico VisDrone.
- Test nel dominio fisico (stampa, CMYK, condizioni di luce reali): l'attacco
  qui valutato è puramente digitale.
- Correggere `_save_checkpoint` per rispettare `--patch-out`.

---

# Letteratura citata

| Fonte | Uso nel progetto |
|---|---|
| Barekatain et al. 2017 | Okutama-Action: dataset aereo, altitudine 10–45m (Fase 7). **Attribuzione obbligatoria in didascalia** per le figure con frame del dataset (CC BY-NC-SA 3.0) |
| Sodhro et al. 2025 | Baseline YOLOv8 outdoor 99.1% confidence; costante 0.710 del simulatore |
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
| LFRAP 2025 | Conferma: pedoni scartati per dimensione da vista aerea. ⚠ riferimento completo da verificare prima del `.bib` |
| Adversarial patch attacks against aerial imagery object detectors 2023 | Loss su feature intermedie. ⚠ riferimento completo da verificare prima del `.bib` |
| Everingham et al. 2010 | Standard IoU=0.5 (PASCAL VOC), da cui ci discostiamo |
| Yu et al. 2020 | "Scale Match for Tiny Person Detection" — IoU permissivo |
| Shao et al. 2018 | CrowdHuman — convenzione "ignore region" |
| Efron & Tibshirani 1993 | Bootstrap appaiato |

DOI/URL completi in `config.py`.



## FASE 8 — Stesura Cap. 4, verifiche sul codice, correzione della stratificazione

Sessione dedicata alla scrittura del Capitolo 4 (Metodo). Tutte le voci sotto
sono state verificate leggendo il codice sorgente o eseguendo strumentazione,
non ricostruite da note precedenti.

---

### 8.1 Correzioni da applicare alle sezioni precedenti di questo documento

⚠ **§7.8 — il finding sulla dipendenza dalla scala è un artefatto.**
La misura riportata come `[VERIFICATO]` (bucket 100–150 px al 96.8%) era
priva di condizione di controllo. Eseguita la condizione di controllo con lo
stesso criterio, il bucket risulta **già al 97.1% senza patch**. Il 96.8%
misurava il limite di localizzazione del rilevatore, non l'effetto
dell'attacco.

Va corretto anche il «controllo di coerenza indipendente» a 1280 con patch
VisDrone (bucket 150+ px al 97.2%): stessa assenza di controllo, e la regione
di scala è la medesima (150 px @1280 ≡ 112 px @960). Non erano due misure
indipendenti dello stesso fenomeno, erano due misure dello stesso artefatto.

Entrambe le misure erano su **dati Okutama**. La stratificazione non è mai
stata eseguita su dati VisDrone.

⚠ **§7.8 — il confound IoU dichiarato `[APERTO]` è stato risolto.**
`stratify_by_size.py` ora attribuisce l'esito per bersaglio mediante
corrispondenza IoU con soglia `IOU_IGNORE_THRESHOLD = 0.3`, importata da
`simulator.py`. Aggiunta la condizione di controllo `--no-patch`.

⚠ **Fase 2, tabella configurazioni** — la voce sul learning rate come
"ininfluente" non risulta da alcuno sweep documentato. Le sei configurazioni
variano loss, aggregazione, accumulo e scheduler, non l'LR. Declassare a
`[IPOTESI]` o rimuovere: `config.py` riporta solo una motivazione a priori
(passo ridotto per evitare la saturazione della sigmoide).

⚠ **Riferimenti bibliografici** — `xi_2025_lfrap` e `tang_2023`, marcati
«⚠ riferimento completo da verificare», sono stati verificati e sono corretti.
Rimuovere l'avvertenza.

---

### 8.2 Stratificazione per scala — misura definitiva

`[VERIFICATO]` `stratify_by_size.py` con matching IoU e condizione di
controllo. Dati Okutama-val, 960×960, patch addestrata su Okutama,
soglia di confidenza 0.50.

| Bucket altezza (@960) | Controllo | Attacco | Delta | n |
|---|---|---|---|---|
| 60–100 px | 70.0% | 97.0% | **+27.0 pp** | 19921 |
| 100–150 px | 97.1% | 100.0% | +2.9 pp | 618 |
| 150 px+ | — | — | — | 0 |

**Lettura corretta.** L'effetto dell'attacco è nel bucket 60–100 px, che
contiene il 97% dei bersagli. Il bucket 100–150 px è in saturazione: la
condizione di controllo è già al 97.1%, non resta margine misurabile, e
n=618 è piccolo. Il bucket 150 px+ è vuoto per costruzione (a 960 le altezze
sono 0.75× quelle a 1280).

**Verifica di coerenza interna** `[VERIFICATO]`: il delta per bersaglio con
matching IoU (**+27.0 pp**) coincide con il delta a livello di fotogramma
misurato indipendentemente dal report bootstrap (**+28.0 pp**, 0.497 → 0.777).
Due unità di analisi diverse, due criteri di attribuzione diversi, stesso
risultato. È l'argomento più solido disponibile per la difesa.

---

### 8.3 Frazione di iterazioni utili — causa accertata

`[VERIFICATO]` per strumentazione diretta del ciclo di ottimizzazione, con
contatori su tutti i punti di uscita. Okutama, 960, 200 iterazioni:

```
{'no_bbox': 0, 'mask_empty': 0, 'no_targets': 0, 'tactical_zero': 75, 'ok': 125}
```

Somma esatta 200. **Unica causa di scarto: il filtro tattico**, ossia
fotogrammi in cui nessun bersaglio supera i 60 px e il peso di rilevanza
dimensionale è nullo. Zero scarti per maschera spaziale vuota o bbox
degenere.

Un'ipotesi formulata in sessione — interazione fra risoluzione del canvas e
*stride* del rilevatore, con bersagli troppo sottili per contenere il centro
di una cella — è stata **falsificata** da questa misura (`mask_empty = 0`).
Non va riportata in tesi.

Frazione utile misurata: 62.5% su questo campione, coerente con il 55.5%
ricavato sul run completo (1110 aggiornamenti su 2000 attesi).

---

### 8.4 Run di training analizzato — identificazione e proprietà

`[VERIFICATO]` `training_metrics.json` (senza suffisso) è il run **Okutama,
8000 iterazioni**. Identificato per quattro vie concordi: posizione fuori da
`archive_fase6_visdrone/`, 1110 aggiornamenti contro i 1250 del run VisDrone
finale, budget 8000 contro 5000, e frazione di iterazioni utili (55.5%)
corrispondente alla densità di bersagli di Okutama e non a quella di VisDrone
(15.1%).

⚠ La curva di loss del run VisDrone finale (Fase 2, config #5) resta
**perduta**: sovrascritta prima dell'adozione della convenzione di naming.
Non rigenerabile senza rilanciare il training. La figura della curva di loss
in tesi può essere solo Okutama.

**Proprietà misurate sul run:**

| Grandezza | Valore |
|---|---|
| Aggiornamenti attesi / realizzati | 2000 / **1110** (55.5%) |
| Learning rate finale osservato | 0.004726 (`eta_min` atteso: 0.001) |
| Media mobile loss, minimo | 0.7574 all'aggiornamento 840 |
| Media mobile loss, finale | 0.7583 |
| Norma del gradiente, intervallo | 6.8×10⁻⁴ – 3.0×10⁻² |
| Soglia di troncamento del gradiente | 1.0 — **mai raggiunta** |
| $\mathcal{L}_{TV}$ osservata | 0.0466 – 0.0630 |
| Contributo TV alla norma del gradiente | mediana 2.6×10⁻⁶ |

Lo scarto del learning rate è stato verificato ricalcolando la formula del
coseno: con `T_max = 2000` e $t = 1110$ il valore atteso è 0.0047263,
coincidente alla sesta cifra con quello loggato.

**Decisione: non rilanciare il training.** Le sei configurazioni di Fase 2
convergono in una banda di cinque punti nonostante varino anche lo
scheduler; la media mobile della loss è piatta dall'aggiornamento 840; il
troncamento del gradiente non interviene mai. Il collo di bottiglia non è il
programma del passo. Lo scarto va dichiarato fra le limitazioni.

---

### 8.5 Divario fra obiettivo di ottimizzazione e metrica di valutazione

`[VERIFICATO]` per inversione della definizione della loss,
$\mathcal{L} = -\ln(1 - \bar{c} + \varepsilon)$:

$$\bar{c}: 0.5465 \longrightarrow 0.5312 \qquad (\Delta = -1.5\ \text{pp})$$

Contro un delta di evasione di **+28.0 pp** sullo stesso dominio. Il rapporto
è circa 1:18.

`[VERIFICATO]` dal codice, due delle tre cause: la loss agisce sulla media
delle top-20 celle mentre il rilevamento dipende dal solo massimo
superstite alla NMS; la loss è un valore atteso su 16 trasformazioni (scale
0.4–0.9, rotazioni ±15°) mentre la valutazione avviene su una singola
realizzazione.

`[INTERPRETAZIONE]` la terza: la soglia di confidenza è una funzione a
gradino, quindi una traslazione piccola di una distribuzione addensata
attorno alla soglia produce una variazione grande del tasso di
attraversamento. Promuovibile a `[VERIFICATO]` loggando la distribuzione
delle confidenze post-NMS attorno alla soglia in una passata di
`--eval-report`, senza ritraining.

---

### 8.6 Difetti di implementazione accertati, da dichiarare

`[VERIFICATO]` **La patch valutata è l'ultima, non la migliore.**
`optimize_universal` ritorna la patch dell'ultimo aggiornamento;
`cli.py` la salva con `torch.save(results["patch"], patch_out)` dove
`patch_out` è per default `BEST_PATCH_FILE`, sovrascrivendo il checkpoint
migliore salvato durante il training. Differenza in media mobile: 0.0009,
trascurabile. In tesi va scritto «al termine del budget di ottimizzazione»,
non «patch migliore».

`[VERIFICATO]` **L'arresto anticipato non è mai intervenuto.** Con
`WINDOW=100` e `PATIENCE=1000` su 1110 aggiornamenti disponibili, la
condizione è strutturalmente irraggiungibile (100 per iniziare a misurare +
1000 senza miglioramento = 1100 su 1110). Da presentare come salvaguardia
contro la divergenza, non come criterio di selezione del modello.

`[VERIFICATO]` **`ihl_overrides` in `fusion_decision.py` è dead code.**
Il blocco `if action == "ENGAGE" and not ihl` non può eseguirsi, perché
`action` diventa `"ENGAGE"` solo se `ihl` è già vero. Il comportamento a
valle è corretto (un vincolo violato porta comunque ad ALERT per l'altro
ramo), ma il contatore resta a zero. Fix disponibile: separare la decisione
per soglia dal veto IHL; output bit-identico, il contatore diventa
osservabile.

⚠ **Commenti obsoleti in `config.py`**: riga 54 (`-> 1250 update reali`) e
riga 59 (`Scalato per 1250 update potenziali (5000/4)`) si riferiscono a un
budget non più in uso.

---

### 8.7 Discrepanza fra sorgenti di risultati — da risolvere prima del Cap. 5

⚠ `vision_metrics.json` riporta R1 pre 0.5012 / post 0.2200, R2 0.9699.
`full_report_okutama_960_stride27.json` riporta 0.5033 / 0.2233 / 0.9604.
Probabile differenza di sottoinsieme o di *stride* fra le due esecuzioni.
**Il dato da citare in tesi è quello del report decorrelato a n=527.**

---

### 8.8 Formulazione della TV loss — equazione corretta

`[VERIFICATO]` il codice implementa la forma **anisotropa in norma L1 con
medie separate**, non quella isotropa di Thys et al.:

```python
dx = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
dy = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
return dx + dy
```

L'equazione in tesi è stata scritta di conseguenza. La motivazione riportata
(assenza di radice quadrata, quindi assenza di singolarità del gradiente
dove la superficie è localmente costante) è una proprietà della forma
adottata, non una scelta documentata nel codice.

---

### 8.9 Bibliografia — audit completo

`[VERIFICATO]` 84 entry, zero chiavi duplicate, zero titoli duplicati, zero
chiavi citate e mancanti.

Correzioni applicate:

| Chiave | Problema | Stato |
|---|---|---|
| `gonzalez_2021_etip` | DOI errato e titolo troncato; vol. 58, art. 102715 | corretto anche in `config.py` |
| `sodhro_2025` | Autori incompleti, titolo troncato | corretto |
| `adhikari_2020` | Ordine autori arXiv con DOI SPIE | convertito a `@misc` |
| `asili_2025_gentprm` | Primo autore errato: **Asılı, Harika** (non "Asili, Hamed"); titolo inventato | corretto, `@inproceedings` ACIT 2025 |

---

### 8.10 Vincoli formali del correlatore — documento acquisito

`LineeGuidaTesiLatex.pdf` letto integralmente. Vincoli rilevanti applicati al
Cap. 4: profondità massima 1.1.1; sottosezioni sotto la pagina accorpate
(§4.2 portata da sei a quattro); punto decimale e non virgola; nessun `\\`
per andare a capo; notazione matematica (vettori minuscolo grassetto,
matrici maiuscolo non corsivo, scalari minuscolo corsivo) — `\mathbf{P}`
sostituito con `\mathrm{P}`; ogni figura con didascalia, label `fig:` e
`\ref` nel testo; ogni equazione con label `eq:` e spiegazione testuale dei
simboli.

**Indicazione sulla lunghezza** (comunicazione del correlatore): circa 50–60
pagine da copertina a copertina per una magistrale, 100 considerate
eccessive. Caso in esame complicato dalla co-relazione: circa 40 pagine di
inquadramento concordate con l'altro relatore. Questione sottoposta al
correlatore, in attesa di riscontro.

**Indicazione sul data fusion**: media pesata considerata adeguata; i metodi
bayesiani «spesso finiscono per essere medie pesate». Conseguenza per la
stesura: non enfatizzare l'aspetto bayesiano del Livello 3 oltre quanto
l'implementazione giustifica.

---

### 8.11 Stato del Capitolo 4

Scritte e revisionate: §4.1 (architettura, tre figure TikZ), §4.2 (modello di
minaccia, parametrizzazione, robustezza fisica, vincoli di piattaforma),
§4.3 (funzione obiettivo, top-K, divario obiettivo/metrica), §4.4 (metriche),
§4.5 (insiemi di dati), §4.6 (protocollo statistico).

Creato `tikz_common_styles.tex` con palette desaturata e larghezze
centralizzate. ⚠ Nota di manutenzione: righe vuote all'interno di
`\tikzset{}` producono `Paragraph ended before \pgfkeys@addpath was
complete`; nel file sono neutralizzate con `%`.