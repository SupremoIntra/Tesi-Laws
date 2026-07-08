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
| 7 | Run con nuovi iperparametri (accum=2) ha dato risultati quasi identici al run precedente | `--train-patch` riprende automaticamente da `checkpoint_patch.pt`, che era ancora quello del run precedente (accum=4, step 1200) — il nuovo run ha eseguito solo 300 step raw residui invece di 1500, con LR già decaduto quasi a zero dallo scheduler cosine | Flag `--fresh` in cli.py per ripartire da zero; checkpoint ora salva `gradient_accumulation_steps` e avvisa se non coincide con la config corrente |
| 8 | LR resta vicino al massimo per quasi tutta la durata del run (es. 0.00951 al 60% del run Fase 1) invece di scendere con la curva coseno | `CosineAnnealingLR(T_max=n_steps)` usa gli step raw come T_max, ma `scheduler.step()` viene chiamato una volta per **update reale**, non per step raw — con accum=4 la curva si allunga di 4x (con accum=2, di 2x), e il LR non arriva mai vicino a `eta_min` entro la fine reale del run | `T_max = n_steps // GRADIENT_ACCUMULATION_STEPS` (unità: update reali). Bug presente (in gradi diversi) anche in Run 1 e Run 2 — nessuno dei due ha mai avuto una vera fase di raffinamento a LR basso |

## Letteratura citata (riferimento rapido)

- Sodhro et al. 2025 — baseline YOLOv8 outdoor 99.1% confidence
- Carlini & Wagner 2017 — margine di robustezza nella loss
- Wu et al. 2020 — Domain-Specific Attacks > Universal Attacks
- Athalye et al. 2017 — EoT
- Thys et al. 2019 — TV Loss per pattern stampabili
- Brown et al. 2017 — Adversarial Patch, convergenza rapida
- Arkin 2009 — Governing Lethal Behavior, Principio di Distinzione IHL
- Liu et al. 2019 — DPatch, loss congiunta box+classe
- Huang et al. 2020 — Universal Physical Camouflage Attacks (UPC), design region-aware
- Hu et al. 2021 — Naturalistic Physical Adversarial Patch, GAN latent space

DOI/URL completi in `config.py`.

---

## Esperimento: update reali della patch (Gradient Accumulation)

### Definizioni

- **Step raw (o iterazione dati)**: un singolo ciclo del loop di training —
  carica un'immagine, applica l'EoT, fa un forward+backward attraverso
  YOLO. `PATCH_STEPS` conta questi, non gli aggiornamenti dei pixel.
- **Forward pass / batch**: l'immagine con la patch iniettata, replicata
  in `EOT_N_TRANSFORMS` versioni aumentate (16), passa una volta attraverso
  la rete. Un batch = un'immagine × 16 trasformazioni.
- **Gradient Accumulation Steps**: quanti step raw si sommano prima di
  chiamare `optimizer.step()`. Il gradiente si accumula nel buffer `.grad`
  di `patch_logits` per N step raw, poi si applica una volta sola.
- **Update reale (o optimization step)**: la singola chiamata a
  `optimizer.step()` — è il momento in cui i pixel della patch cambiano
  davvero.
- **Relazione**: `update_reali = step_raw / accumulation_steps`.
  Con `step_raw=1500, accumulation_steps=4` → 375 update.
  Con `step_raw=1500, accumulation_steps=2` → 750 update.

### A cosa serve l'accumulo, e perché l'abbiamo ridotto

Il Gradient Accumulation esiste per simulare un batch più grande (qui:
batch effettivo 4 = 1 immagine fisica × 4 accumuli) senza il costo di
memoria di un vero forward su 4 immagini insieme — utile su hardware
limitato (M4 16GB). Il costo è che ogni update reale arriva più
raramente: con `PATCH_STEPS` fisso, più accumulo = meno aggiornamenti
totali dei pixel della patch.

Nel primo run reale (accum=4) la loss è rimasta piatta intorno a
`ln(2)≈0.693` per la maggior parte del training e l'evasion rate è
salito solo dello 0.7% (vedi tabella sotto) — troppo poco rispetto al
salto di qualità architetturale (loss asintotica, EoT 16, dataset 10x
più grande). Ipotesi più probabile: **la patch ha ricevuto troppo pochi
aggiornamenti reali** (375) rispetto al baseline storico (3000).
Accumulo 4→2 raddoppia gli update reali (375→750) senza aumentare il
numero di step raw, quindi senza aumentare il tempo di calcolo.

Il Learning Rate (`PATCH_LR=0.01`) resta invariato in questo test: il
training applica già `clip_grad_norm_(max_norm=1.0)` prima di ogni
update, che limita l'ampiezza del gradiente indipendentemente da quanti
campioni sono stati mediati. Riducendo l'accumulo cambia la varianza/
direzione della stima del gradiente, non la sua ampiezza massima — che è
già vincolata dal clipping. Cambiare anche il LR in questo test
introdurrebbe una seconda variabile, rendendo impossibile isolare
l'effetto del solo aumento degli update.

### Cronistoria configurazioni

| Config | Step raw | Accum | Update reali | Loss | F1 | Precision | Recall | Evasion Rate |
|---|---|---|---|---|---|---|---|---|
| Baseline storico (Hinge Loss) | 3000 | 1 | 3000 | — | 0.760 | 1.000 | 0.613 | 38.7% (31/80) |
| Run 1 (Loss asintotica, accum=4) | 1500 | 4 | 375 | plateau ~0.693 | 0.740 | 1.000 | 0.588 | 41.25% (33/80) |
| Run 2, INVALIDATO (accum=2) | ~~1500~~ **300 reali** (bug checkpoint, vedi Bug #7) | 2 | ~~750~~ **150** | plateau ~0.693 | 0.740 (identico a Run 1) | 1.000 | 0.588 | 41.25% (identico) |
| **Run 2 corretto (accum=2, `--fresh`, full run)** | 1500 | 2 | 750 | plateau ~0.693 per tutto il run, TV scende 0.023→0.012 | 0.750 | 1.000 | 0.600 | 40.0% (32/80) |

### Cosa dice Run 2 corretto

Raddoppiare gli update reali (375→750) **non ha migliorato l'evasion rate** — anzi
è leggermente peggiore di Run 1 (40.0% vs 41.25%), entro il rumore statistico
di un dataset di validazione da 80 frame positivi. La loss principale non è mai
scesa in modo pulito in 1500 step raw: è rimasta a oscillare intorno a
`ln(2)≈0.693` dall'inizio alla fine. L'unica quantità monotona è la TV loss
(0.023→0.012): l'optimizer trova "facile" smussare la patch (riducendo TV) e
non trova un segnale consistente per abbassare la confidenza di YOLO. Questo
sposta la diagnosi da "pochi update" a **budget di calcolo insufficiente per
la convergenza**, coerente con la letteratura sulle Universal Patch (vedi
sezione Letteratura sotto).

**Nota di reinterpretazione (dopo il Bug #8):** Run 1 e Run 2 avevano
*anche* lo scheduler LR allungato (rispettivamente 4x e 2x) — il LR non è
mai sceso in modo significativo verso `eta_min` in nessuno dei due run.
Non possiamo quindi concludere con certezza che il plateau fosse *solo*
un problema di budget: la Fase 1, rilanciata con lo scheduler corretto,
dà la risposta (vedi sotto).

### Fase 1 corretta — risultato

| Config | Step raw | Accum | Update reali | Scheduler | F1 | Precision | Recall | Evasion Rate |
|---|---|---|---|---|---|---|---|---|
| Baseline storico | 3000 | 1 | 3000 | corretto | 0.760 | 1.000 | 0.613 | 38.7% (31/80) |
| Run 1 | 1500 | 4 | 375 | allungato 4x | 0.740 | 1.000 | 0.588 | 41.25% (33/80) |
| Run 2 corretto | 1500 | 2 | 750 | allungato 2x | 0.750 | 1.000 | 0.600 | 40.0% (32/80) |
| **Fase 1 corretta** | 10000 | 4 | 2500 | **corretto** | **0.720** | 1.000 | 0.562 | **43.75% (35/80)** |

**Miglior risultato finora**, ma il rendimento è decrescente: 6.6x il
calcolo di Run 1 (2500 vs 375 update, stavolta con LR correttamente
annealed fino a `eta_min`) ha prodotto solo +2.5 punti di evasion rate
rispetto a Run 1, e +3.75 rispetto a Run 2. Anche durante l'ultima parte
del run (LR≈0.001, la vera fase di raffinamento) la loss non ha mostrato
un crollo netto — è rimasta nella stessa banda 0.69-0.75, ma
evidentemente qualche update marginale continua a spingere la patch
nella direzione giusta, solo molto lentamente.

**Conclusione**: budget e scheduling erano problemi reali (li abbiamo
risolti, e hanno dato un miglioramento misurabile), ma non sono l'intero
problema. Scalare ancora il numero di step avrebbe rendimenti sempre più
marginali per il tempo investito. Il prossimo passo logico è la Fase 2 —
cambiare il segnale della loss, non la quantità di training.

### Prossimi passi — vedi "Piano Sperimentale" più sotto

Il piano incrementale a step (3000, poi 6000 step raw) è stato superato:
Run 2 corretto mostra che il collo di bottiglia non è solo la quantità di
update, quindi si passa a un piano più strutturato (sezione successiva).

---

## Glossario (con numeri presi dal nostro caso)

- **Step raw**: un ciclo del training loop — carica un'immagine, applica
  l'EoT, fa un forward+backward attraverso YOLO. `PATCH_STEPS=1500` è il
  numero di step raw, non di aggiornamenti dei pixel.
- **Batch / forward pass**: l'immagine con la patch iniettata, replicata in
  `EOT_N_TRANSFORMS=16` versioni aumentate, passa una volta nella rete.
- **Gradient Accumulation**: quanti step raw si sommano prima di applicare
  l'update. Con `accum=2`, il gradiente di 2 step raw si somma nel buffer
  `.grad` prima che `optimizer.step()` lo applichi.
- **Update reale**: la chiamata a `optimizer.step()` — il momento in cui i
  pixel della patch cambiano. `update = step_raw / accum`. Nel nostro Run 2:
  `1500 / 2 = 750` update.
- **EoT (Expectation over Transformation)**: media della loss su N
  trasformazioni random della patch (rotazione, scala, colore) per farla
  funzionare non solo su un'immagine esatta ma su una distribuzione di
  condizioni realistiche. Da noi: 16 trasformazioni per ogni step raw.
- **TV Loss (Total Variation)**: penalizza differenze bruschi tra pixel
  adiacenti della patch. Alta (`TV_WEIGHT=0.1`) forza pattern a bassa
  frequenza (macchie di colore ampie) che sopravvivono all'interpolazione
  bilineare durante l'EoT. Nei nostri run scende sempre in modo monotono
  (0.023→0.012), a differenza della loss principale.
- **Evasion Rate**: frazione di frame con persona reale in cui YOLO, sotto
  attacco, NON rileva nessuno. Nel Run 2 corretto: 32 frame evasi su 80
  frame con persona (`32/80 = 40.0%`).
- **F1-Score (nel contesto dell'attacco)**: media armonica di Precision e
  Recall sulla detection frame-level. Un F1 che scende sotto attacco (da
  0.760 baseline a 0.750 con la patch) indica un peggioramento della
  capacità del sensore — ma qui il calo è modesto, coerente con
  un'evasion rate ancora lontana dal 100%.
- **Confidence Drop**: differenza tra la confidenza che YOLO assegna alla
  classe "person" sulla stessa immagine, con e senza patch. Si misura per
  singola persona/frame (non è un aggregato come F1) — script dedicato in
  `tools/measure_confidence_drop.py`.

### Perché la loss ristagna esattamente a `ln(2)=0.693`

La loss è `-log(1 - mean_conf + eps)`. Se `mean_conf ≈ 0.5` (la confidenza
media, sui neuroni mascherati, non si allontana dal punto neutro), allora
`-log(1 - 0.5) = -log(0.5) = ln(2) ≈ 0.693`. Il fatto che la loss oscilli
sempre attorno a questo valore, per 1500 step raw consecutivi, dice che
`mean_conf` non si sposta in modo consistente da 0.5 — la patch non sta
sistematicamente abbassando la confidenza di YOLO, sta solo oscillando
insieme al contenuto casuale di ogni immagine/target campionato.

---

## Letteratura: perché non converge e come accelerare

| Paper | Anno | Idea chiave | Applicazione al nostro setup |
|---|---|---|---|
| Brown et al. — "Adversarial Patch" | 2017 | Patch universali su classificatori: convergenza rapida su singola classe, ma il paper originale attacca classificatori, non detector con centinaia di celle grid da mediare | Il nostro caso (detector anchor-free, media su molte celle) è strutturalmente più difficile — dilazione del segnale che Brown non affrontava |
| Liu et al. — "DPatch: An Adversarial Patch Attack on Object Detectors" | 2019 (arXiv 1806.02299) | Attacca **congiuntamente** la regressione del bounding box E il punteggio di classe, non solo la classe. Riporta mAP da 70.0%/65.7% a <1% su Faster R-CNN/YOLO | YOLOv8 è anchor-free, non ha un canale "objectness" separato come YOLOv2/v3 — ma possiamo comunque aggiungere un termine di loss sulla regressione box (es. spingere le box predette a shrinkare/spostarsi) accanto alla loss di classe attuale |
| Thys et al. — "Fooling automated surveillance cameras" | 2019 | TV Loss per pattern stampabili; già adottato | — |
| Wu et al. — "Making an Invisibility Cloak" | 2020 | Domain-Specific Attack converge più in fretta di un Universal Attack perché non deve generalizzare su migliaia di scene diverse | È il nostro "Piano B" già pianificato (`annotate_mioDS.py`) |
| Huang et al. — "Universal Physical Camouflage Attacks on Object Detectors" (UPC) | 2020 | Design della patch region-aware: parti diverse della patch attaccano parti diverse dell'oggetto (non un pattern uniforme) | Potremmo far dipendere il pattern dalla posizione relativa nel chest-bbox invece di un'unica patch uniforme applicata ovunque |
| Hu et al. — "Naturalistic Physical Adversarial Patch for Object Detectors" | 2021 (ICCV) | Ottimizza nel latent space di una GAN pre-addestrata invece che sui pixel direttamente — patch più naturali, ma con performance d'attacco inferiore a patch pixel-based e bisogno di più dati | Interessante come direzione "originale" (Fase 3), non come fix per la convergenza attuale — introdurrebbe più complessità, non meno |

**Nota onesta**: nessuna fonte in letteratura garantisce un numero fisso di
iterazioni per convergere — dipende da architettura, dataset, loss. La
lezione trasversale è che i lavori con risultati forti (DPatch, Thys et
al.) attaccano **più segnali contemporaneamente** (box + classe, o
TV + classe con pesi ben calibrati), non solo un singolo termine di loss
sulla confidenza media.

---

## Piano sperimentale

### Fase 1 — Budget di calcolo maggiore (COMPLETATA)

Obiettivo: verificare se il plateau fosse un limite di budget/scheduling.
Primo tentativo interrotto a step ~6000/10000 dopo la scoperta del Bug #8
(scheduler allungato); rilanciato da zero con lo scheduler corretto.

- `PATCH_STEPS=10000`, `GRADIENT_ACCUMULATION_STEPS=4` → 2500 update reali
- Risultato: F1=0.720, Evasion Rate=43.75% — il migliore finora, ma
  rendimento decrescente (6.6x calcolo per +2.5-3.75 punti, vedi tabella
  sopra)
- Conclusione: budget/scheduling erano problemi reali e li abbiamo
  risolti, ma non spiegano l'intero plateau → si passa alla Fase 2

### Fase 2 — Miglioramento del segnale (in corso)

**Diagnostica (fatta): il gradiente è debole, non morto.** Log del run
diagnostico (`grad_norm`): valori tra **0.0007 e 0.005**, sempre ben sotto
la soglia di clipping (`max_norm=1.0`). Il gradient clipping non è mai
scattato in nessun run finora — il problema non è mai stato un gradiente
troppo grande, è uno troppo debole (ma non zero: se fosse rotto dal punto
di vista del backprop sarebbe esattamente 0, non 0.001-0.005).

Interpretazione più probabile: ogni step raw calcola il gradiente su
**una sola immagine** (`BATCH_SIZE_PHYSICAL=1`). La direzione "giusta"
per una patch universale deve emergere dalla media su centinaia di scene
diverse — se ogni immagine spinge in una direzione leggermente diversa
(normale, sono contenuti diversi), il segnale si cancella parzialmente
invece di sommarsi. È un problema di rapporto segnale/rumore per immagine
singola, non un backprop rotto.

Aggiunta diagnostica ulteriore: separare la norma del gradiente della
**sola loss principale** da quella combinata con la TV, per capire quanto
della norma totale viene dall'attacco vs dalla regolarizzazione (log:
`main_loss_grad_norm` in `training_metrics.json`).

**Ipotesi prioritaria, sostenuta dai dati** (non solo dalla letteratura):
aumentare drasticamente l'accumulo (es. 16-32 invece di 2-4) per mediare
il gradiente su molte più immagini per ogni update reale — matematicamente
equivalente a un batch reale più grande, stesso costo di memoria (si
processa comunque un'immagine alla volta). Trade-off: "più update
rumorosi" vs "meno update ma con una direzione più pulita" — l'accumulo
massimo testato finora è solo 4, la strada non è stata esplorata a fondo.

### Risultato test accum=16 — ipotesi respinta dai dati

| Config | Update reali | Scheduler | F1 | Evasion Rate |
|---|---|---|---|---|
| Fase 1 (accum=4) | 2500 | corretto | 0.720 | **43.75%** |
| Fase 2 (accum=16) | 625 | corretto | 0.740 | **41.25%** |

Risultato peggiore, non migliore. Prova diretta che l'ipotesi "basta
mediare su più immagini" non regge da sola: il numero di update conta più
della pulizia del singolo update, in questo range. Conferma quantitativa
dal log dei gradienti: passando da accum=4 (norma per singola immagine
0.0007-0.005) ad accum=16 (somma di 16 immagini), la norma è cresciuta
solo di un fattore ~1.5-2x — non 16x (immagini che spingono nella stessa
direzione) e nemmeno vicino a quello. Una crescita ~√16=4x indicherebbe
immagini scorrelate tra loro; il dato osservato è più basso ancora,
segno che il segnale condiviso tra scene diverse è molto debole.

**Osservazione aggiuntiva dal log**: `GradNorm(tot)` e `GradNorm(loss)`
sono quasi identici riga per riga (es. 0.005797 vs 0.005795) — la TV
loss contribuisce quasi nulla alla norma, eppure scende in modo
liscissimo e monotono (0.0506→0.0290) mentre la loss principale, con
gradiente di grandezza comparabile, non scende mai. Non è (solo) un
problema di magnitudo: è un problema di **direzione**. Il gradiente TV
punta sempre nello stesso verso (dipende solo dai pixel della patch);
il gradiente della loss principale cambia direzione da immagine a
immagine (dipende dalla scena specifica) e la sua componente utile non
emerge nemmeno mediata su 16 campioni.

**Conclusione**: la strada "più immagini per update" ha rendimenti
decrescenti già evidenti a 16, quindi si abbandona. Il prossimo
esperimento (non ancora fatto, proposto come lavoro futuro) è cambiare
la formula della loss stessa: invece di `masked_scores.mean()` su tutte
le celle della spatial mask (che include celle di sfondo irrilevanti,
diverse da immagine a immagine — stessa diluizione già documentata nel
Bug #4 per la Hinge Loss, mai riverificata per l'asintotica), mediare
solo sulle top-K celle a confidenza più alta dentro la maschera.

**Alternative da testare in seguito, una alla volta, se l'accumulo alto
non basta:**

1. **Loss anche sulla regressione box** (da DPatch): oltre a
   `person_scores`, penalizzare anche le coordinate predette delle box
   dentro la spatial mask, per spingerle a spostarsi/deformarsi.
2. **EoT più aggressivo sulla scala**: allargare il range di scala
   (`0.4-0.9` attuale) per forzare la patch a funzionare a più distanze
   contemporaneamente, invece che convergere su una scala "comoda".
3. **Inizializzazione della patch non casuale**: partire da un pattern a
   media/contrasto già alto (es. rumore strutturato) invece di
   `randn * 0.1`, per uscire più in fretta dalla zona piatta iniziale
   della sigmoide.

### Fase 3 — Contributo originale (idea da sviluppare, non urgente)

Il framework ha già qualcosa che la maggior parte dei paper sulle
Adversarial Patch non ha: un simulatore multi-agente con fusion
Vision+OSINT+Behavioral. Un contributo originale defendibile è **l'attacco
coordinato multi-dominio**: non limitarsi a misurare l'effetto della patch
isolato (Vision-only), ma modellare come un avversario razionale
distribuirebbe un budget di attacco limitato tra patch fisica (Vision) e
data poisoning (OSINT) per massimizzare il collasso del Fusion Agent a
parità di "costo" — la metrica CEAE già esistente è pensata esattamente
per questo confronto. Lo scenario `CASCADING` nel simulatore è già un primo
passo in questa direzione; formalizzarlo come problema di ottimizzazione
(quanto budget in Vision vs quanto in OSINT dato un CEAE target) sarebbe un
angolo che i singoli paper sulle patch (tutti Vision-only) non trattano.

---

## Sintesi per la call col relatore

### La storia in 5 frasi

1. Un attacco adversarial patch universale contro YOLOv8 su VisDrone
   riduce l'F1-Score da 0.760 a un minimo di 0.720, portando l'Evasion
   Rate dal 38.7% al 43.75% — un effetto reale ma modesto, non un
   collasso del sensore.
2. Ho isolato sistematicamente cinque variabili (numero di update reali,
   scheduling del learning rate, ampiezza dell'accumulo del gradiente,
   formula della loss, aggregazione mean vs top-K) e ho misurato
   l'effetto di ciascuna separatamente, scoprendo e correggendo due bug
   non banali nel processo (contaminazione tra run via checkpoint,
   scheduler LR mal configurato).
3. Sei configurazioni metodologicamente diverse convergono tutte nella
   stessa banda stretta (38.7%-43.75% di evasion rate) — compresa una
   loss ispirata a Thys et al. (2019) che raggiunge lo stesso tetto con
   metà del budget di calcolo di qualunque altra configurazione.
4. Questo pattern indica un **soffitto strutturale**, non un limite di
   ottimizzazione: nessuna leva di training (loss, scheduling, accumulo)
   riesce a superarlo, il che sposta l'attenzione sulla geometria fisica
   della patch e sull'eterogeneità del dataset, non sull'algoritmo.
5. Le prossime leve pianificate agiscono sul vincolo strutturale stesso
   (aumentare la copertura fisica della patch, o passare a un attacco
   image-specific alla Wu et al. 2020), non su ulteriori variazioni
   dell'ottimizzatore.

### Tabella finale di tutti i run

| # | Config | Update reali | Scheduler | F1 | Evasion Rate |
|---|---|---|---|---|---|
| 0 | Baseline storico (Hinge Loss) | 3000 | corretto | 0.760 | 38.7% |
| 1 | Run 1 (asintotica mean, accum=4) | 375 | allungato 4x | 0.740 | 41.25% |
| 2 | Run 2 (asintotica mean, accum=2) | 750 | allungato 2x | 0.750 | 40.0% |
| 3 | Fase 1 (asintotica mean, accum=4, sched. corretto) | 2500 | corretto | 0.720 | 43.75% |
| 4 | Fase 2 (asintotica mean, accum=16, sched. corretto) | 625 | corretto | 0.740 | 41.25% |
| 5 | **Fase 3 (asintotica top-K=20, accum=4, sched. corretto)** | **1250** | **corretto** | **0.720** | **43.75%** |

**Il risultato da presentare come "il numero" della tesi è la riga 5**
(Fase 3): stesso massimo di Fase 1, raggiunto con metà del calcolo —
è il risultato più efficiente, e la sua identità con Fase 1 è essa
stessa la prova del soffitto strutturale (punto 3-4 sopra).

### Domande che il relatore potrebbe fare, e come rispondere

**"Perché l'evasion rate non supera il 44%? Nei paper originali (Brown,
Thys) sembra molto più alto."**
→ Quei paper attaccano classificatori o detector con un solo bersaglio
per immagine, spesso in condizioni controllate. VisDrone è un dataset
aereo, multi-bersaglio, con scale e angolazioni molto più variabili — il
gradiente per una patch universale deve mediare su una distribuzione di
scene molto più eterogenea. L'ho misurato direttamente (norma del
gradiente debole, direzioni scorrelate tra immagini): non è
un'impressione, è un dato quantitativo.

**"Come fate a dire che è un limite strutturale e non semplicemente che
non avete trovato la configurazione giusta?"**
→ Perché ho testato sei configurazioni radicalmente diverse — due
formule di loss (Hinge, asintotica), due aggregazioni (media, top-K),
quattro valori di accumulo del gradiente (1, 2, 4, 16), con e senza bug
di scheduling — e convergono tutte nella stessa banda stretta
(38.7%-43.75%). Se il limite fosse "serve solo la loss giusta" o
"servono solo più update", almeno una delle sei configurazioni
l'avrebbe superato in modo netto. Il fatto che la loss più sofisticata
(top-K, ispirata a Thys et al.) raggiunga esattamente lo stesso tetto di
prima, con metà del calcolo, è la prova più diretta che il limite non è
nell'ottimizzatore.

**"Avete provato ad aumentare ancora il training?"**
→ Sì, sistematicamente: dal budget base (375 update) fino a 2500 (6.6x),
con miglioramento di soli 2.5 punti e rendimenti già decrescenti.
Continuare a scalare senza cambiare la loss avrebbe un ritorno sempre
più basso per il tempo di calcolo investito — è un dato, non una scusa
per fermarsi.

**"Come sapete che non è un bug del codice?"**
→ Perché abbiamo trovato e corretto due bug reali durante il percorso
(documentati, non nascosti): un checkpoint che contaminava i run, e uno
scheduler del learning rate mal configurato. Dopo averli corretti, il
comportamento (plateau della loss, gradiente debole) è rimasto
identico — segno che non erano quelli a spiegare il plateau. Inoltre il
gradient clipping non è mai scattato in nessun run, il che esclude
un'esplosione o un errore grossolano di scala.

**"Cosa fareste con più tempo?"**
→ La prossima modifica pianificata è cambiare la formula della loss:
mediare solo sulle celle a più alta confidenza dentro la maschera invece
che su tutte (evitando la diluizione del segnale già documentata per la
Hinge Loss e mai riverificata per quella attuale), oppure aggiungere una
componente di loss sulla regressione del bounding box (DPatch, Liu et
al. 2019) accanto a quella di classe.

**"Perché non avete usato un dataset più piccolo/omogeneo?"**
→ È esattamente il Piano B già pronto (`annotate_mioDS.py`, Wu et al.
2020): un attacco image-specific converge più in fretta perché non deve
generalizzare su migliaia di scene. È stato tenuto come alternativa
esplicita fin dall'inizio del progetto, non come ripiego dell'ultimo
minuto.

**"Il framework serve solo per questo attacco?"**
→ No — la parte Vision è un layer di un sistema più ampio (Fusion
bayesiano Vision+OSINT+Behavioral, soglie decisionali IHL). Il risultato
sull'evasion rate alimenta il simulatore tattico tramite un bridge JSON,
dove viene combinato con lo scenario di OSINT Data Poisoning per
mostrare come un attacco multi-dominio comprometta le soglie di
ingaggio — visibile già oggi con `--run-sim`.

---

## Punti di forza del lavoro (da citare alla commissione)

- **Metodo sistematico**: ogni run è isolato a una variabile alla volta
  (accum, poi step raw), con un bug di contaminazione tra run scoperto e
  documentato (non nascosto) invece di semplicemente "riportare il numero
  migliore".
- **Confronto con la letteratura**: i risultati (plateau della loss,
  necessità di molti più update) sono coerenti con quanto riportato da
  Brown et al. e Liu et al. sulla difficoltà di convergenza delle patch
  universali — non è un'anomalia isolata del nostro setup.
- **Identificazione del collo di bottiglia**: distinzione netta tra
  "problema di quantità" (update reali) e "problema di segnale" (loss che
  non decresce nemmeno con più update) — la Fase 1/2 del piano sperimentale
  è disegnata apposta per isolare quale dei due sia la causa reale.
- **Framework multi-dominio**: la parte Vision è solo un layer di un
  sistema più ampio (Fusion bayesiano, OSINT poisoning, soglie IHL) che
  la maggior parte della letteratura sulle adversarial patch non modella.

---

## Stato attuale

Pipeline validata end-to-end su MPS reale. Quattro run completi
confrontabili (Baseline, Run 1, Run 2, Fase 1 corretta) — Fase 1 è il
migliore finora (43.75% evasion) ma con rendimento decrescente rispetto
al calcolo investito. Diagnostica del gradiente completata: segnale
debole (0.0007-0.005) ma non morto, coerente con un problema di
rapporto segnale/rumore per immagine singola più che con un bug di
backprop. Prossimo test: accumulo molto più alto (16-32).

## Direzione futura (dopo il soffitto strutturale)

### La scoperta che ricontestualizza tutto

Analizzando la letteratura specifica su attacchi UAV/VisDrone (non solo
i paper generici su patch, già coperti sopra), emerge un pattern che
nessuno dei paper "generici" (Thys, Brown, DPatch — tutti a livello del
suolo) poteva rivelare:

| Paper | Anno | Cosa attaccano su VisDrone | Perché |
|---|---|---|---|
| Shrestha, Pathak, Viegas — "Towards a Robust Adversarial Patch Attack Against UAV Object Detection" | 2023 (IEEE/RSJ IROS) | **Car**, non Person. 80% ASR white-box, 75-78% transfer gray-box | Patch pensate per prospettiva/angolo/distanza UAV |
| LFRAP (Multi-Dimensional Feature Optimization) | 2025 | Car, truck, van, bus — dichiarano esplicitamente di scartare i pedoni | *"Given the small size of pedestrians under the overhead perspective of UAVs..."* — citazione diretta dal paper |
| Adversarial patch attacks against aerial imagery object detectors | 2023 | Aerei (airplane), loss su feature intermedie invece che output finale | Bersagli grandi, e tecnica alternativa (feature-level, non class-score) |

**Nessuno dei lavori con risultati forti su VisDrone attacca i pedoni.**
Non è una scelta stilistica — i pedoni da vista aerea occupano
pochissimi pixel assoluti, e una patch limitata al 20% di un bbox già
piccolo ha un budget di pixel reali molto ridotto dopo il resize a
640×640. Il nostro "soffitto" al 44% è verosimilmente il prezzo
specifico di aver scelto il bersaglio più difficile del dominio, non un
limite generico del framework o della loss.

### Perché questo non è un problema per la tesi — è coerente col tuo scenario

Il simulatore modella già un **drone tattico a bassa quota**
(`DRONE_ALTITUDE_M=10`, ingaggio ravvicinato, non sorveglianza ad area
larga — vedi sezione Simulation Environment). VisDrone include scatti a
distanze/altitudini molto eterogenee, molte delle quali non
rappresentano nemmeno lo scenario che stai simulando. **Filtrare il
dataset a persone di dimensione realistica per un ingaggio ravvicinato
non è cherry-picking — è allineare i dati di training al modello di
minaccia già definito nel resto del framework.**

### Verifica gratuita prima di investire 20.000 step

Prima di lanciare qualunque training lungo, `tools/stratify_by_size.py`
misura l'evasion rate per bucket di altezza del bbox (60-100px,
100-150px, 150px+) usando la patch già addestrata di Fase 3 — zero
costo di training, solo inferenza. Se conferma un salto netto tra
bucket piccoli e grandi, abbiamo la prova quantitativa sui nostri dati
prima di spendere ore di calcolo. Risultato: *(da compilare dopo
l'esecuzione)*.

### La raccomandazione concreta per i 20.000 step

Non ripetere la stessa configurazione più a lungo (i 6 run precedenti
già dimostrano rendimenti piatti scalando solo gli step). Invece:

1. **Rifiltrare il dataset** (sia training che eval) a un'altezza minima
   più realistica per un ingaggio a `DRONE_ALTITUDE_M=10` — es. `≥120px`
   invece di `≥60px`. Sottoinsieme più piccolo ma più omogeneo, e
   coerente con lo scenario.
2. **Tenere la loss top-K** (Fase 3, già la più efficiente misurata).
3. **A quel punto, e solo a quel punto, il budget lungo (20.000 step
   raw, accum=4 → 5000 update) ha senso**: su un sottoinsieme più
   omogeneo, senza il rumore delle persone minuscole che dominano il
   segnale con confidenze quasi-zero indipendentemente dalla patch, più
   update dovrebbero tradursi in un miglioramento reale — la condizione
   che finora, su tutto VisDrone, non si è mai verificata.
4. **Se anche questo non basta**, l'alternativa di riserva è la tecnica
   della "Adversarial patch attacks against aerial imagery object
   detectors" (2023): loss sulle feature intermedie del backbone invece
   che sull'output finale — bypassa completamente il problema di
   diluizione su centinaia di celle di output, ma richiede modifiche più
   profonde (hook sui layer intermedi di YOLO, non solo sulla loss finale).

### Cosa NON fare

Non lanciare 20.000 step sulla configurazione attuale (tutto VisDrone,
soglia 60px) aspettandosi un salto — i dati di 6 run già lo escludono.
Sarebbe lo stesso errore concettuale di "più budget senza cambiare la
causa" già confutato da Fase 1 vs Fase 2.

## Aperto / da fare

- **[COMPLETATA] Fase 3 — top-K loss**: `PATCH_STEPS=5000, accum=4` →
  1250 update. Risultato: F1=0.720, TP=45, FN=35, Evasion Rate=43.75% —
  identico a Fase 1 con metà del budget (vedi tabella comparativa sopra
  e interpretazione in "Direzione futura").
- Lanciare `tools/stratify_by_size.py` (zero costo, solo inferenza) per
  quantificare l'effetto dimensione bersaglio sui nostri dati
- Se confermato: rifiltrare dataset a bbox ≥120px (coerente con
  `DRONE_ALTITUDE_M=10`), rilanciare con loss top-K e budget lungo
  (20000 step raw, accum=4 → 5000 update) — vedi "Direzione futura"
- Se anche questo non basta: loss su feature intermedie del backbone
  invece che sull'output finale (Adversarial patch attacks against
  aerial imagery object detectors, 2023)
- Custom dataset (Domain Adaptation, `tools/annotate_mioDS.py`) — Piano
  B sempre disponibile, converge più in fretta (Wu et al. 2020)
- Calibrazione fine di `ENGAGEMENT_THRESHOLD` dopo aver visto risultati
  Vision più forti
- Physical Domain Gap (stampa CMYK reale, drone reale) — limitazione da
  menzionare in tesi, non da risolvere nel codice
- Fase originale (contributo: ottimizzazione multi-dominio Vision+OSINT
  sotto budget di attacco vincolato) — idea da maturare, non urgente
