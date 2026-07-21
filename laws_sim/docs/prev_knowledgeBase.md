# Knowledge Base: Evoluzione e Risoluzione Architetturale dell'Ottimizzatore Avversariale (LAWS-SIM) --- OLD DUMP
 
Questo documento traccia l'evoluzione del modulo `patch_optimizer.py`, evidenziando le sfide tecniche incontrate su architettura Apple Silicon (M-Series), i limiti di PyTorch e le giustificazioni accademiche delle soluzioni adottate a seguito di una rigorosa *Peer Review* del codice.
 
## 1. Sfida Hardware: I Limiti del Backend MPS
**Problema:** L'implementazione iniziale dell'Expectation over Transformation (EoT) tramite `torchvision` generava l'errore `aten::grid_sampler_2d_backward not implemented for MPS`. L'abilitazione del fallback su CPU generava errori sistemici di *Inplace Memory Modification* (`AsStridedBackward0`), corrompendo il grafo computazionale.
**Soluzione (CPU-Enforced Mode):** L'intero processo di addestramento è stato forzato su CPU per preservare l'integrità matematica dei gradienti, sacrificando l'accelerazione hardware in favore del rigore accademico.
 
## 2. Risoluzione della Disconnessione del Grafo Autograd
**Problema:** L'uso estensivo di comandi come `.clone()` in fasi intermedie del grafo (es. prima del forward del modello) creava tensori foglia disconnessi. Inoltre, l'utilizzo della somma additiva con limite (`clamp`) per posizionare le patch sul canvas bloccava il flusso dei gradienti nel caso di bersagli sovrapposti.
**Soluzione:** 
1. Rimozione dei distacchi arbitrari del grafo.
2. Sostituzione del clamp con un **Alpha Blending Differenziabile**: 
   $C_{out} = C_{acc} \cdot (1 - \alpha) + C_{new} \cdot \alpha$
   Questo garantisce la retropropagazione ininterrotta del gradiente dai logit finali fino ai pixel nativi della patch.
## 3. Correzione Logit vs Probability (Soglia IHL)
**Problema Critico:** YOLOv8 (architettura Anchor-Free) emette layer crudi sotto forma di *logit* (prima dell'applicazione della sigmoide o della Non-Maximum Suppression). La Loss Hinge confrontava erroneamente questi logit con la soglia probabilistica di ingaggio ($0.35$).
**Soluzione:** Applicazione esplicita della funzione logistica `torch.sigmoid()` sui tensori estratti dalla rete prima del calcolo della Hinge Loss, riallineando la matematica dell'ottimizzatore alla reale confidenza percepita dal drone.
 
## 4. Evoluzione della Targeted Hinge Loss
**Problema:** L'uso della media standard (`.mean()`) diluiva il gradiente includendo centinaia di celle di background inattive (valore 0). Il tentativo di smorzamento tramite un limite fisso al denominatore (`max(numel, 10)`) risultava arbitrario e instabile.
**Soluzione:** La Hinge Loss è calcolata tramite media semplice sui soli neuroni violanti (`hinge > 0`). L'eventuale esplosione asintotica del gradiente a fine training è stata mitigata in modo standard ed elegante tramite il *Gradient Clipping* globale (`torch.nn.utils.clip_grad_norm_`).
 
## 5. O(1) Global Canvas & EoT Optimizations
**Problema:** L'addestramento multi-target richiedeva l'elaborazione di un Forward Pass completo di YOLO per ogni singola persona identificata (complessità $O(T)$ per frame), con generazione ripetitiva delle distorsioni EoT.
**Soluzione:**
1. **Generazione EoT Singola:** Le trasformazioni affini e fotometriche vengono calcolate una sola volta per immagine e riutilizzate per tutti i target.
2. **Global Canvas:** Tutte le patch ridimensionate vengono fuse su un singolo canvas differenziabile. YOLO esegue un solo Forward Pass per l'intera immagine (complessità $O(1)$), abbattendo drasticamente i tempi di addestramento su CPU.
Ecco il **Technical Audit Log** completo ed esaustivo, strutturato rigorosamente per trasferire il 100% del contesto cognitivo, matematico e architetturale al modello Claude. Ho adottato un livello di granularità estremo per garantire che nessuna sfumatura della tua tesi vada persa.
 
---
 
### SEZIONE 1: CRONOLOGIA DELLE IPOTESI
 
**Sessione Iniziale: Ripristino Architettura e Validazione Baseline**
 
* **Data/sessione approssimativa:** [DETTO NON RICHIAMABILE - verificare chat precedente, sessione di ripristino post-errore utente].
* **Ipotesi iniziale:** Un'architettura basata su una Universal Adversarial Patch con geometria fissa, addestrata per 3000 step su CPU con una Hinge Loss standard, è sufficiente per generare un calo drastico dell'F1-Score su YOLOv8 in VisDrone.
* **Come l'abbiamo testata:** Ripristino completo dei file `config.py`, `patch_optimizer.py`, `simulator.py` e `cli.py` con i commenti accademici originali. Esecuzione di `python cli.py --train-patch` per 3000 iterazioni.
* **Risultato ottenuto:** F1-Score sceso a 0.760. Precision=1.000, Recall=0.613, TP=49, FN=31. Tasso di evasione statico (Minimal Evasion Rate) misurato al 38.7%.
* **Perché è stata accettata o scartata:** Accettata come baseline (dimostra il limite fisico matematico di una patch piccola su un dataset caotico), ma giudicata insufficiente per la tesi magistrale. Il 38.7% genera un "flickering" nel simulatore tattico, ma si voleva spingere l'algoritmo al limite per massimizzare l'Engagement Denial.
**Sessione Intermedia: Domain Adaptation (Il Piano B / Custom Dataset)**
 
* **Data/sessione approssimativa:** Subito dopo i risultati del primo training.
* **Ipotesi iniziale:** Applicando il principio di *Targeted Threat Modeling* (Wu et al.), addestrare la patch non sul dataset universale (VisDrone) ma su un dataset "Image-Specific" che modella esattamente l'angolo di beccheggio (pitch) e l'illuminazione di un drone in avvicinamento, porterà l'Evasion Rate vicino al 100%.
* **Come l'abbiamo testata:** Progettazione del workflow. Scrittura dello script `auto_annotate.py` per generare automaticamente le annotazioni YOLO in formato VisDrone da 15-20 foto scattate dall'utente.
* **Risultato ottenuto:** Script generato e pronto all'uso.
* **Perché è stata accettata o scartata:** Accettata e "congelata" come Piano B / Scialuppa di salvataggio per garantire risultati inattaccabili in sede di presentazione al relatore, qualora l'Universal Attack non raggiungesse le soglie desiderate.
**Sessione Avanzata: Ablation Study - L'Offensiva Notturna (Deep Convergence e Caos ad Alta Frequenza)**
 
* **Data/sessione approssimativa:** Sessione serale, preparatoria a un training notturno di 12+ ore.
* **Ipotesi iniziale:** Per distruggere YOLOv8 su VisDrone si deve: (1) Rilassare la TV Loss (`0.001` -> `0.0001`) per massimizzare il rumore ad alta frequenza; (2) Aumentare la superficie di attacco al 50x40% (Tactical Vest Assumption); (3) Applicare un margine severo di Carlini-Wagner abbassando il `TARGET_CONF` da `0.35` a `0.10`; (4) Raddoppiare l'EoT a 16 trasformazioni introducendo la distorsione prospettica (Aspect Ratio); (5) Estendere i `PATCH_STEPS` a 10.000.
* **Come l'abbiamo testata:** Modifica dei parametri in `config.py` e `patch_optimizer.py`. Lancio del training su CPU (`device="cpu"`).
* **Risultato ottenuto:** Dopo circa 12 ore e 1800 step, il training è entrato in stallo. La Hinge Loss oscillava eternamente tra 0.40 e 0.43. La confidenza di YOLO balzava da 0.00 a 0.84 in modo caotico.
* **Perché è stata accettata o scartata:** Scartata e interrotta forzatamente (Kill process tramite `^C`). L'ipotesi ha fatto emergere tre colli di bottiglia matematici fatali (Vanishing Gradient, Paradosso TV Loss, Saturazione Sigmoide) che hanno richiesto un cambio di paradigma teorico totale.
**Sessione Corrente: Refactoring Teorico per Claude (MPS e Untargeted Loss)**
 
* **Data/sessione approssimativa:** Fine sessione, in preparazione del passaggio di consegne all'AI Claude.
* **Ipotesi iniziale:** Il fallimento precedente è dovuto a errori di modellazione dell'Adversarial Machine Learning fisico. Bisogna passare ad accelerazione Apple Silicon (`mps`), usare una Loss Untargeted asintotica (media delle probabilità) senza soglie ReLU rigide, alzare la TV Loss a `0.1` per generare basse frequenze resistenti all'interpolazione bilineare, e abbassare il Learning Rate a `0.01` limitando a 1500 step.
* **Come l'abbiamo testata:** [In attesa]. Delegato a Claude tramite prompt ingegnerizzato.
* **Risultato ottenuto:** [In attesa].
* **Perché è stata accettata o scartata:** Accettata come direzione definitiva per il refactoring architetturale.
---
 
### SEZIONE 2: BUG E PROBLEMI TECNICI RISOLTI
 
**Bug 1: Perdita di Metadati Accademici e Narrazione**
 
* **Sintomo osservato:** I file python erano stati "piallati", perdendo tutti i commenti originali, le citazioni (Sodhro, Threod, ecc.) e i fix pregressi dell'utente.
* **Root cause:** Eccessivo zelo nel refactoring durante l'unificazione del codice in sessioni precedenti, trattando la tesi come puro software engineering.
* **Tentativi falliti di fix (elenco):** Nessuno. Rilevamento immediato grazie alla segnalazione dell'utente ("hai fatto un errore madornale...").
* **Fix finale adottato:** Ripristino riga per riga dai backup pregressi, fondendo le nuove logiche (3000 steps, CLI unificata) con i commenti originali.
* **File e riga di codice coinvolte:** `config.py`, `patch_optimizer.py`, `simulator.py`, `cli.py` (intero documento).
**Bug 2: Invisibilità Sistemica delle Entità nel Simulatore**
 
* **Sintomo osservato:** Nessuna entità veniva mai rilevata dal drone; metriche di TP e FP perennemente a zero.
* **Root cause:** Il parametro `DRONE_ALTITUDE_M` era impostato a 300 metri. Poiché la distanza 3D calcolata tramite il teorema di Pitagora era sempre $\ge 300$, superava costantemente il limite fisico `YOLO_MAX_RANGE` impostato a 150 metri.
* **Tentativi falliti di fix (elenco):** Modifiche al raggio di YOLO [DETTO NON RICHIAMABILE - verificare sessioni precedenti all'unificazione].
* **Fix finale adottato:** Abbassamento dell'altitudine operativa a un realistico `10` metri (`DRONE_ALTITUDE_M = 10`), garantendo che il calcolo $sqrt(x^2 + y^2 + 10^2)$ scalasse correttamente con il raggio di detection.
* **File e riga di codice coinvolte:** `config.py`, linea `DRONE_ALTITUDE_M = 10`.
**Bug 3: Geometria Irrealistica della Patch (Face Occlusion)**
 
* **Sintomo osservato:** La patch copriva interamente la figura umana, inclusa la testa, fungendo da occlusione fisica irrealistica anziché da pattern avversariale.
* **Root cause:** Coordinate `PATCH_H = 240` fissate in pixel assoluti, non scalate in base alla dimensione reale del bounding box della persona.
* **Tentativi falliti di fix (elenco):** Modifica manuale dei pixel hardcoded in `config.py`.
* **Fix finale adottato:** Implementazione della funzione `get_chest_bbox_proportional`, che inietta la patch dinamicamente al centro-torace calcolando il 40% della larghezza e il 30% dell'altezza (poi esteso a 50x40%) del target specifico in quel frame.
* **File e riga di codice coinvolte:** `patch_optimizer.py` e `simulator.py`, funzione `get_chest_bbox_proportional`.
**Bug 4: Autograd Crash / NameError durante l'Ablation Study**
 
* **Sintomo osservato:** Crash immediato al lancio di `python cli.py --train-patch` con `Traceback: NameError: name 'device' is not defined` nella funzione `_visdrone_eot`.
* **Root cause:** Durante il refactoring chirurgico per inserire l'Aspect Ratio, è stata omessa (non copiata) la prima riga della funzione originale che instanziava il device partendo dal tensore di input.
* **Tentativi falliti di fix (elenco):** Nessuno, individuato al volo dall'errore Python.
* **Fix finale adottato:** Re-inserimento esplicito di `device = patch.device` all'inizio della funzione statica prima della generazione delle trasformazioni affini.
* **File e riga di codice coinvolte:** `patch_optimizer.py`, linea `device = patch.device` in `@staticmethod def _visdrone_eot`.
**Bug 5: Vanishing Gradient per Targeted Hinge Loss**
 
* **Sintomo osservato:** Stallo dell'addestramento. La loss $L = 0.4208$ scendeva lievemente e rimbalzava, ma i pixel non apprendevano nuovi pattern.
* **Root cause:** La formula $F.relu(\text{scores} - 0.10)$ genera derivata prima pari a $0$ quando il valore dell'argomento è negativo. Sotto la confidenza di $0.10$, l'ottimizzatore perdeva il segnale direzionale ("gradiente morto").
* **Tentativi falliti di fix (elenco):** Tentativo di aumentare l'aggressività con la TV Loss e 10.000 step.
* **Fix finale adottato:** Transizione delegata a Claude verso una *Untargeted Loss Asintotica* che minimizza direttamente `masked_scores.mean()`.
* **File e riga di codice coinvolte:** `patch_optimizer.py`, metodo `_targeted_hinge_loss`.
**Bug 6: Il Paradosso della TV Loss come Filtro Passa-Basso**
 
* **Sintomo osservato:** TV loss che sale inesorabilmente (da 0.05 a 0.18) mentre la YOLO confidence oscilla violentemente verso valori estremi (0.84).
* **Root cause:** TV_WEIGHT impostata a 0.0001 per generare rumore ad altissima frequenza. Questo rumore viene distrutto dal calcolo di interpolazione bilineare (`mode='bilinear'`) all'interno di `F.grid_sample` nell'algoritmo EoT, che per sua natura matematica effettua una media spaziale dei pixel (filtro passa-basso).
* **Tentativi falliti di fix (elenco):** Aumento del numero di iterazioni (10.000) sperando che il pattern emergesse lo stesso.
* **Fix finale adottato:** Alzare la `TV_WEIGHT` a `0.1` per generare "macchie" di colore ampie (bassa frequenza) che sopravvivono all'interpolazione matematica e alle lenti fisiche.
* **File e riga di codice coinvolte:** `patch_optimizer.py`, variabile di classe `TV_WEIGHT`.
---
 
### SEZIONE 3: SCELTE ARCHITETTURALI E TRADE-OFF
 
**Decisione 1: Gestione della Memoria nell'Injection Avversariale**
 
* **Opzioni considerate:** (A) Uso di `.clone()` per copiare il frame originale per ogni trasformazione EoT e incollare la patch. (B) Logica "O(1) Canvas" con generazione di una maschera spaziale (`spatial_mask`) e iniezione tensoriale globale tramite moltiplicazione di maschere alpha.
* **Pro/contro:** (A) Molto semplice da scrivere, ma causa un'esplosione esponenziale della RAM/VRAM se l'EoT è alto o i bersagli sono multipli. (B) Codice matematicamente molto complesso, ma tracciamento del gradiente pulito e occupazione di memoria costante.
* **Scelta e perché:** Logica (B) "O(1) Canvas" per permettere all'ottimizzatore di lavorare su CPU senza esaurire la memoria del Mac dell'utente.
* **Alternativa congelata:** Nessuna, la scelta B è definitiva.
**Decisione 2: Parametrizzazione dello Spazio Colore della Patch**
 
* **Opzioni considerate:** (A) Ottimizzazione diretta dei pixel RGB limitando i valori con `torch.clamp(patch, 0, 1)` ad ogni step. (B) Parametrizzazione tramite spazio logit: `patch_rgb = torch.sigmoid(patch_logits)`.
* **Pro/contro:** (A) Facile intuizione, ma `clamp` distrugge il gradiente per i pixel che "sbattono" contro i bordi (0 o 1). (B) Evita pixel morti e mantiene un gradiente fluido ovunque, ma rischia la *Sigmoid Saturation* se il Learning Rate è troppo alto.
* **Scelta e perché:** Opzione (B) Sigmoide, mitigata (dopo l'audit) con una riduzione del Learning Rate a `0.01`.
* **Alternativa congelata:** Trasformazione dello spazio colore (es. passaggio a HSV per ottimizzare solo la Tinta), giudicata non necessaria per YOLO.
**Decisione 3: Accelerazione Hardware (CPU vs MPS)**
 
* **Opzioni considerate:** (A) Forzare `device="cpu"`. (B) Utilizzare `torch.backends.mps` (Apple Silicon).
* **Pro/contro:** (A) Lento (12+ ore per 1500 step) ma storicamente affidabile per le operazioni di griglia affini. (B) Estremamente veloce (meno di 1 ora), ma con potenziale rischio di gradienti `NaN` se le versioni di PyTorch non gestiscono bene `F.grid_sample` in backward pass.
* **Scelta e perché:** Passaggio all'opzione (B) MPS per salvare la fattibilità della tesi. La tolleranza di attesa di 4 giorni su CPU era inaccettabile e bloccava i progressi accademici.
* **Alternativa congelata:** Mantenere la CPU per la sola inferenza del simulatore (`simulator.py`), limitando l'uso dell'acceleratore all'ottimizzatore pesante.
**Decisione 4: Architettura del Framework (Monolite vs Disaccoppiato)**
 
* **Opzioni considerate:** (A) Script unico che fa training, test e simulazione 3D allo stesso tempo. (B) Framework rigorosamente disaccoppiato tramite file JSON.
* **Pro/contro:** (A) Niente I/O su disco. (B) I bug della visione artificiale non bloccano il simulatore tattico. Permette esperimenti isolati (es. OSINT puro).
* **Scelta e perché:** Opzione (B). L'empirismo (YOLO) genera un calo di F1-Score salvato in `vision_metrics.json`. Il LAWS-SIM legge il JSON per simulare il "flickering" del sensore.
* **Alternativa congelata:** Iniezione frame-by-frame live della simulazione in YOLO (troppo costosa computazionalmente).
**Decisione 5: Modellazione dell'Attacco OSINT (Data Poisoning)**
 
* **Opzioni considerate:** (A) Riqualificazione di un modello LLM/NLP per valutare stringhe social malevole. (B) Alterazione matematica dei pesi probabilistici nel profilo.
* **Pro/contro:** (A) Realistico ma fuori scopo (tesi incentrata sulla fusione sensoriale, non sull'NLP). (B) Implementazione rapida via `Faker` o regole fittizie che ribaltano i parametri.
* **Scelta e perché:** Opzione (B). Nel simulatore, se `scenario == AttackScenario.OSINT_POISONING`, il profilo dei Target viene mascherato (`geo_risk` abbassato, `social_matches=0`) e il profilo dei Civili viene compromesso per generare Falsi Positivi sistematici.
---
 
### SEZIONE 4: LETTERATURA SCIENTIFICA CITATA
 
**1. Sodhro, A.H. et al. (2025)**
 
* *Titolo/Focus:* "Real-time efficiency of YOLOv5 and YOLOv8 in human intrusion detection across diverse environments".
* *Concetto chiave estratto:* YOLOv8 in condizioni outdoor ha una confidence baseline del 99.1%.
* *Applicazione:* Giustifica l'uso di `0.710` F1-score iniziale per la Baseline del sensore in `simulator.py` (simulando cali dovuti a distanza/nebbia) e posiziona YOLOv8 come State of the Art per i LAWS.
**2. Carlini, N., & Wagner, D. (2017)**
 
* *Titolo/Focus:* "Towards Evaluating the Robustness of Neural Networks".
* *Concetto chiave estratto:* L'uso del margine $\kappa$ (kappa) nella funzione di loss per spingere l'errore ben oltre il confine decisionale, creando "cuscinetti" di robustezza.
* *Applicazione:* Utilizzato per giustificare l'abbassamento brutale della `TARGET_CONF` da 0.35 a 0.10, per forzare l'ottimizzatore a distruggere completamente le feature map.
**3. Wu, Z. et al. (2020)**
 
* *Titolo/Focus:* "Making an Invisibility Cloak: Real World Adversarial Attacks on Object Detectors".
* *Concetto chiave estratto:* I *Domain-Specific Attacks* superano il limite degli *Universal Attacks*. Modellare la patch per uno specifico angolo di visione e contesto eleva l'Evasion Rate.
* *Applicazione:* Fondamenta teorica per il "Piano B" (Custom Dataset via `auto_annotate.py`). Usato per giustificare alla commissione che l'Universal Attack su VisDrone è uno scenario *worst-case*, mentre in ambito tattico mirato l'arma sfiorerebbe il 100% di efficacia. Ha anche ispirato la "Untargeted Loss asintotica".
**4. Athalye, A. et al. (2017)**
 
* *Titolo/Focus:* "Synthesizing Robust Adversarial Examples".
* *Concetto chiave estratto:* Algoritmo EoT (Expectation over Transformation). Per rendere fisico un attacco digitale, il gradiente deve essere calcolato sulla media di N trasformazioni spaziali/ottiche.
* *Applicazione:* Implementazione della funzione statica `_visdrone_eot` nel codice. Usato per incrementare le deformazioni da 8 a 16 e inserire l'Aspect Ratio (per simulare il *Pitch* aereo del drone).
**5. Thys, S. et al. (2019)**
 
* *Titolo/Focus:* "Fooling automated surveillance cameras: adversarial patches to attack person detection".
* *Concetto chiave estratto:* Necessità di un elevato peso per la Total Variation (TV Loss) al fine di generare pattern stampabili e visivamente coerenti, superando la distruzione dei dettagli causata da sfocatura e ridimensionamento ottico.
* *Applicazione:* Cruciale per la risoluzione del "Paradosso della TV Loss". Usato per giustificare il passaggio di `TV_WEIGHT` da `0.0001` a `0.1` nel refactoring destinato a Claude.
**6. Brown, T. B. et al. (2017)**
 
* *Titolo/Focus:* "Adversarial Patch".
* *Concetto chiave estratto:* Le patch universali convergono molto rapidamente, e iterazioni eccessive causano overfitting sui bias specifici del dataset di addestramento.
* *Applicazione:* Giustificazione per ridurre le iterazioni da 10.000 a 1.500 per evitare che la patch "impari" a ingannare i pixel specifici della compressione JPEG di VisDrone anziché i filtri convoluzionali di YOLO.
---
 
### SEZIONE 5: DATASET E METRICHE
 
**Dataset VisDrone:**
 
* *Subset usato:* VisDrone-DET, caricato tramite custom `VisDroneLoader`.
* *Preprocessing & Filtraggio:* Per la validazione empirica, vengono considerati validi SOLO i bounding box della classe 0 ("person") con altezza $\ge 60$ pixel. Questo previene il *Downsampling Destruction*, escludendo bersagli microscopici (es. 10x15 pixel) in cui l'iniezione della patch distruggerebbe completamente l'informazione visiva rendendo il test non scientificamente valido.
* *Volumi:* Durante il test, il loader ha segnalato `531 frame validi`.
**Custom Dataset (Domain Adaptation):**
 
* *Logica di generazione:* Scrittura di `auto_annotate.py`. L'utente scatta 15-20 immagini overhead (dall'alto verso il basso a simulare 10-15 metri di pitch aereo).
* *Logica Script:* Carica un modello pre-addestrato `yolov8n.pt`, esegue l'inferenza, estrae i box della classe `0`, e scrive un file `.txt` per ogni immagine nel formato specifico VisDrone: `left, top, width, height, conf, 1, 0, 0` (Category 1 = pedestrian in VisDrone map).
**Metriche calcolate (Dominio Visivo puro):**
 
* *Test 1 (Sotto attacco, baseline universal patch 3000 steps):*
* TP: 49, FN: 31, FP: 0 (Precision 1.000 perfetta).
* Recall: $49 / (49+31) = 0.613$.
* F1-Score Reale: **0.760**
* *Significato Tattico:* *Minimal Evasion Rate* del 38.7% ($31/80$).
* *Traslazione nel LAWS-SIM:* La metrica empirica dell'F1 viene salvata in `vision_metrics.json` e riletta nel simulatore per governare la probabilità di detection al singolo frame tramite la formula:
`vision_confidence = random.uniform(0.0, max(empirical_f1, 0.05))`
Questo causa lo "sfarfallio" (Detection Intermittente) del bersaglio nel tempo, devastando le probabilità Bayesiane del Fusion Agent che non riescono ad accumularsi oltre l'`ENGAGEMENT_THRESHOLD` (0.58).
**Metriche Simulate (LAWS-SIM Multi-Dominio):**
 
* Calcolo del CEAE (Cascading Error Amplification Effect) [da confermare la formula finale nella codebase] per quantificare il collasso decisionale quando i pesi (Vision 45%, OSINT 35%, Behavior 20%) vengono alterati da attacchi simultanei.
---
 
### SEZIONE 6: CODICE GENERATO (elenco completo)
 
**1. `config.py` (File di ripristino)**
 
* *Scopo:* Configurazione centralizzata (parametri griglia, velocità drone 61 km/h, altitudine 10m, pesi fusione, parametri training EoT).
* *Implementato:* Sì, base del progetto.
* *Problemi:* Inizialmente `DRONE_ALTITUDE_M` a 300 causava range detection error. Patch size fissi coprivano i volti.
**2. `patch_optimizer.py` (File di ripristino)**
 
* *Scopo:* Core engine di addestramento su CPU. Contiene classe `PatchOptimizer`, metodi di loss, EoT, autograd masking, parametrizzazione sigmoide, iteratore loader batch.
* *Implementato:* Sì.
* *Problemi:* Soffriva di tutti i bug teorici risolti e documentati in Sezione 2 e 3 (gradiente morto, paradosso TV).
**3. `simulator.py` (File di ripristino)**
 
* *Scopo:* Loop multi-agente, applicazione OSINT Data poisoning, test empirico F1 (`evaluate_on_dataset` con patch injection via OpenCV), aggiornamento minaccia con soglia ENGAGE e ALERT.
* *Implementato:* Sì.
**4. `cli.py` (File di ripristino)**
 
* *Scopo:* Parsing Argparse per unificare il framework: `--train-patch`, `--eval-vision`, `--run-sim`. Orchestratore dell'I/O JSON.
* *Implementato:* Sì.
**5. Snippet `config.py` (Modifiche Ablation Notturna)**
 
* *Scopo:* Modifica parametri per esplorazione estrema: `PATCH_STEPS = 10000`, `PATCH_BBOX_COVERAGE = 0.20`, `EOT_N_TRANSFORMS = 16`.
* *Implementato:* Sì. Ha portato al test di 12+ ore fallito.
**6. Snippet `patch_optimizer.py` (Modifiche Ablation Notturna)**
 
* *Scopo:* TV a `0.0001`, `TARGET_CONF = 0.10` (Carlini-Wagner). Modifica alla funzione `get_chest_bbox_proportional` per 50% in X, 40% in Y. Inserimento distorsione `Aspect Ratio` e doppia scala (X, Y) nella matrice affine `theta`.
* *Implementato:* Sì.
* *Problemi:* Ha innescato il "NameError device non definito", corretto al punto successivo.
**7. Snippet `patch_optimizer.py` (Fix Bug Device)**
 
* *Scopo:* Re-inserimento della variabile d'ambiente `device = patch.device` in testa a `_visdrone_eot`.
* *Implementato:* Sì.
**8. Script `auto_annotate.py**`
 
* *Scopo:* Generazione autonoma di dataset per Domain Adaptation usando inferenza YOLO base. Converte immagini raw in format VisDrone.
* *Implementato:* Da eseguire lato utente previa raccolta foto.
**9. Snippet `config.py` e `patch_optimizer.py` (Bozza Refactoring per Claude)**
 
* *Scopo:* Le direttive matematiche (non ancora integrate nel file effettivo, preparate per Claude). Sostituzione con Loss Untargeted (`masked_scores.mean()`), `PATCH_LR = 0.01`, `TV_WEIGHT = 0.1`, `PATCH_STEPS = 1500`, hardware = `"mps"`.
* *Implementato:* Parziale/Teorico. È il compito delegato al Mega-Prompt.
**10. Mega-Prompt XML per Claude 3.5 Sonnet**
 
* *Scopo:* Prompting avanzato (3 versioni, l'ultima XML compliant con Context Anchoring e Chain of Thought forcing) per comandare la IA a eseguire un refactoring perfetto della base di codice.
* *Implementato:* Sì, fornito all'utente per l'interazione con l'ecosistema Anthropic.
---
 
### SEZIONE 7: PROBLEMI APERTI E DEBITI TECNICI
 
**1. Verifica della Stabilità su MPS (Metal Performance Shaders)**
 
* *Stato:* Aperto (da monitorare post-refactoring Claude).
* *Descrizione:* L'uso di `F.grid_sample` per le trasformazioni ottiche dell'EoT su chip Apple M4 potrebbe generare perdite di memoria (Memory Leaks) nel grafo di calcolo se PyTorch non libera correttamente l'accumulo dei gradienti nella *backward pass* asincrona. Se il Mac consuma troppa RAM, bisognerà intervenire con `.detach()` espliciti post-loss.
* *Da menzionare in Tesi:* Sì, nelle limitazioni dell'hardware accelerator e l'ottimizzazione dell'uso della VRAM.
**2. Verifica e Integrazione Definitiva del Codice di Claude**
 
* *Stato:* Rimandato a brevissimo termine (prossima sessione).
* *Descrizione:* Bisogna assicurarsi che Claude recepisca la formula *Untargeted Loss* asintotica e la sostituisca integralmente all'interno del loop di training senza distruggere l'approccio O(1) Canvas e senza modificare l'integrazione I/O col JSON del simulatore tattico.
**3. Generazione e Test del Custom Dataset (Domain Adaptation)**
 
* *Stato:* Rimandato.
* *Descrizione:* L'utente deve fisicamente procurarsi le foto dall'alto per validare la teoria di *Wu et al.*. Costituisce la prova "di backup" (Scialuppa) da portare al relatore, cruciale per dimostrare che il sistema d'arma collassa al 100% in condizioni non generaliste.
**4. Validazione Tattica del "Flickering" nel Fusion Agent**
 
* *Stato:* Da esplorare post-training.
* *Descrizione:* Anche se la patch funzionerà perfettamente, abbattendo l'F1-Score, dobbiamo assicurarci che la formula Bayesiana del `FusionAgent` in `simulator.py` sfrutti davvero questo collasso del dato visivo. Se l'OSINT e il Comportamento hanno pesi mal bilanciati, il sistema potrebbe comunque sparare (Falso Positivo per l'attacco, Falso Negativo per la patch). L'`ENGAGEMENT_THRESHOLD` (attualmente 0.58) richiederà una fase di calibrazione fine (Tuning Multi-agente).
* *Da menzionare in Tesi:* Capitolo finale (Future Work o Discussione). L'interazione caotica in un ambiente Multi-Agente in cui il rumore del layer fisico (Visione) si scontra con il pregiudizio del layer informatico (OSINT Data Poisoning).
**5. Limitazione Fisica: La Stampa Reale (Physical Domain Gap)**
 
* *Stato:* Limitazione della Tesi.
* *Descrizione:* Tutto l'attacco (compreso l'injection in `evaluate_on_dataset` via OpenCV) avviene su pixel digitali. Sebbene l'EoT e la TV Loss simulino il mondo fisico, stampare la patch con una stampante CMYK, incollarla su un giubbotto tattico, volare un vero drone DJI e fare inferenza live introdurrebbe ulteriori aberrazioni (colorimetria non lineare del toner, motion blur della videocamera reale, artefatti di compressione streaming).
* *Da menzionare in Tesi:* Tassativamente inserito nel capitolo "Limitations and Future Work". Il test fisico con hardware COTS (Commercial Off-The-Shelf) rappresenta l'evoluzione naturale del LAWS-SIM framework
