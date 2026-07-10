# Speech per call

### Hinge Loss vs Loss Asintotica (perché abbiamo cambiato)

La **Hinge Loss** (quella usata nel baseline storico) funziona così:
prende la confidenza che YOLO assegna a "persona" e calcola
`max(0, confidenza - soglia)`. Se la confidenza è già sotto la soglia
(es. sotto 0.10), il risultato è **esattamente zero** — non "piccolo",
proprio zero. E quando una funzione vale zero in un intervallo, la sua
derivata (il gradiente, cioè il segnale che dice all'ottimizzatore "da
che parte muoverti") è zero anche lei. Risultato pratico: appena la
patch porta la confidenza sotto soglia su una cella, l'ottimizzatore
**smette di ricevere segnale da quella cella**, anche se potrebbe
spingerla ancora più giù. Con centinaia di celle diverse che scendono
sotto soglia in momenti diversi, il training rallenta e si blocca in
un plateau — è il problema che chiamiamo "vanishing gradient" (gradiente
che svanisce).

La **Loss Asintotica** (`-log(1 - confidenza + epsilon)`) non ha mai un
tratto piatto: la sua derivata è sempre maggiore di zero, per qualunque
valore di confidenza tra 0 e 1 — si avvicina a zero ma non lo tocca mai
esattamente. Quindi c'è **sempre** un minimo di spinta a scendere
ulteriormente, non importa quanto la confidenza sia già bassa. Ecco
perché l'abbiamo adottata: elimina strutturalmente il problema del
gradiente morto della Hinge Loss.

### Evasion Rate

È la percentuale di frame (fotogrammi), tra quelli che contengono
davvero una persona (di altezza minima 60 pixel, sotto quella soglia
l'immagine è troppo piccola per essere un test valido), in cui YOLO
**sotto attacco** non rileva nessuno. Esempio concreto dal nostro
risultato migliore: su 80 frame con una persona vera, in 35 YOLO non
rileva nulla → 35/80 = 43.75% di evasion rate. Più alto è, meglio
funziona l'attacco (dal punto di vista di chi vuole evadere).

### Gradient Accumulation ("accumulo")

Il training processa un'immagine alla volta (per motivi di memoria).
Invece di aggiornare i pixel della patch dopo *ogni singola* immagine,
si può scegliere di **sommare** i suggerimenti di N immagini di fila, e
applicare l'aggiornamento una volta sola con quella somma. Questo si
chiama accumulo, e N è `GRADIENT_ACCUMULATION_STEPS`. Con N=4: servono 4
immagini prima che i pixel della patch cambino davvero una volta. Il
motivo per farlo è simulare un "batch" più grande (più immagini viste
insieme) senza il costo di memoria di caricarle tutte insieme. Il
rovescio della medaglia: con lo stesso numero totale di immagini
processate, un accumulo più alto significa **meno aggiornamenti reali**
dei pixel.

### EoT (Expectation over Transformation)

Prima di calcolare la loss, la patch viene trasformata in 16 modi
diversi e casuali (ruotata, scalata, cambiata di colore) e la loss si
calcola sulla media di questi 16. Perché: una patch fisica reale (su un
giubbotto) verrà vista dal drone con angolazioni, distanze e luce
diverse ogni volta — se la ottimizzassimo su una condizione sola,
funzionerebbe solo in quella condizione esatta. L'EoT costringe la patch
a funzionare su una *distribuzione* di condizioni plausibili.

### TV Loss (Total Variation)

Penalizza differenze brusche tra pixel adiacenti della patch. Un peso
alto forza la patch verso macchie di colore ampie e uniformi (bassa
frequenza); un peso basso permette rumore ad alta frequenza (texture
molto dettagliata, quasi "statica"). Il motivo per cui vogliamo un peso
alto: il rumore ad alta frequenza viene distrutto dalle trasformazioni
dell'EoT (in particolare l'interpolazione durante scala/rotazione), e
comunque una texture troppo caotica sarebbe difficile da stampare
fisicamente in modo fedele.

### Top-K (l'ultima modifica, quella più efficace)

La loss guarda solo un pezzo dell'immagine (la maschera spaziale,
`spatial_mask`) corrispondente a dove si trova la persona. Dentro quella
zona ci sono centinaia di "celle" (piccole porzioni della griglia
interna di YOLO). Prima calcolavamo la loss sulla **media di tutte**
queste celle — ma la maggior parte sono sfondo, con confidenza già
vicina a zero, che diluiscono il segnale utile. Ora calcoliamo la
**media solo delle 20 celle più confidenti** — quelle vicine al vero
centro della persona, dove YOLO sta davvero "guardando". Risultato:
segnale più concentrato, stesso risultato finale con metà del calcolo.

### Norma del gradiente / Gradient Clipping

Il gradiente è un vettore (una lista di numeri, uno per pixel della
patch) che dice in che direzione muovere ogni pixel. La sua "norma" è
un singolo numero che misura quanto è grande questo vettore nel
complesso — un modo per dire "quanto è forte il segnale di
aggiornamento in questo istante". Il "gradient clipping" è una rete di
sicurezza: se la norma supera una soglia (nel nostro caso 1.0), viene
ridimensionata per evitare aggiornamenti troppo bruschi che
destabilizzerebbero il training. Nei nostri esperimenti questa soglia
**non è mai scattata** — segno che il problema non è mai stato un
gradiente troppo grande, ma uno troppo debole.

### F1-Score (nel contesto dell'attacco)

Combina Precision (quando YOLO dice "persona", ha ragione?) e Recall
(quante persone vere YOLO trova davvero?) in un unico numero. Sotto
attacco, la Precision resta 1.000 in tutti i nostri esperimenti (YOLO
non inventa mai persone che non ci sono), quindi tutto il calo di F1
viene dal Recall che scende (YOLO perde persone vere).

---

## SPEECH

### Slide 1 — Titolo
Niente da spiegare oltre il titolo. Serve solo a inquadrare: framework
multi-agente per testare le vulnerabilità di un sistema d'arma autonomo
letale simulato.

### Slide 2-3 — Pipeline generale
Le 4 fasi del sistema completo: ottimizzazione della patch → validazione
vision → fusione con OSINT/comportamento → decisione finale (ENGAGE/
ALERT/TRACK/IGNORE). Il punto da comunicare: ogni fase produce un
input per la successiva, ma sono **disaccoppiate** — se cambio la patch
non devo ricalcolare tutto il resto a mano.

### Slide 4 — OSINT: spiegazione completa

**Perché esiste questo layer, e come si collega alla patch (la domanda
che ti sei fatto tu):**

La patch attacca un solo canale: la vista (YOLO). Ma un sistema di
sorveglianza/targeting reale — pensa a piattaforme di data fusion come
Palantir, non a una singola telecamera — non decide *solo* guardando
un'immagine. Incrocia anche altri dati: se quella targa è in una
blacklist, se quella zona è ad alto rischio, quante tracce social
sospette ha quella persona. Questo è il livello OSINT (Open Source
Intelligence). **Non si "contrappone" alla patch — è un canale
completamente indipendente.** Un attivista o un civile potrebbe
riuscire a ingannare la telecamera con un pattern fisico (la patch), ma
se il sistema lo identifica comunque tramite metadati (telefono,
social, database governativi), l'evasione visiva da sola non basta a
sfuggire al sistema. Ecco perché il framework modella *entrambi* i
canali: per essere onesti sul fatto che difendersi da un LAWS reale
richiede più che ingannare una telecamera. Lo scenario "Cascading" nel
simulatore è proprio il caso in cui *entrambi* i canali sono sotto
attacco insieme.

Quindi: la confidenza YOLO (vision) e il Threat Score OSINT sono due
numeri diversi, calcolati da dati diversi, che si combinano solo alla
fine nella Fusion bayesiana (45% vision, 35% OSINT, 20% comportamento).
Non è una contraddizione se uno sale e l'altro scende — sono
semplicemente due sensori diversi che il sistema ascolta insieme.

**Scelte di modellazione statistica — spiegate una per una, da zero:**

Il profilo OSINT di ogni entità simulata (target o civile) ha 3
attributi, ognuno generato con una distribuzione statistica diversa
perché ogni attributo ha una natura diversa:

1. **Blacklist targa/watchlist governativa → distribuzione di
   Bernoulli.** Una Bernoulli modella un evento che può essere solo
   Sì/No (0 o 1) — come lanciare una moneta truccata. Qui: "questa
   targa è in blacklist?" è binario, non ha via di mezzo. Impostiamo la
   probabilità di "Sì" all'75% per i target (verosimilmente in lista) e
   all'1% per i civili (falsi positivi rari, come nella realtà).

2. **Geo-rischio della zona → distribuzione Beta.** A differenza della
   Bernoulli, qui il valore non è 0/1 ma un numero continuo tra 0 e 1
   (es. 0.73 di rischio). La distribuzione Beta è fatta apposta per
   modellare probabilità/proporzioni continue, e si può "piegare" verso
   l'alto o verso il basso scegliendo due parametri (alpha, beta). Per
   i target usiamo Beta(8,2), che piega la curva verso valori alti
   (zone di conflitto, ispirato ai dati reali ACLED sui conflitti
   armati); per i civili Beta(2,8), piegata verso il basso (zone
   sicure).

3. **Tracce social/OSINT sospette → distribuzione di Poisson.** Poisson
   modella *quante volte* succede un evento in un intervallo, quando gli
   eventi sono discreti e contati (0, 1, 2, 3...) — tipicamente usata
   per "quante telefonate arrivano in un'ora" o, nel nostro caso,
   "quante tracce social sospette ha questa persona". Ha un solo
   parametro (lambda = il valore medio atteso): per i target lambda=12
   (in media 12 tracce, ispirato a casi reali di foto/menzioni in
   contesti sospetti), per i civili lambda=1 (quasi nessuna traccia).

4. **Inferenza Bayesiana → il Threat Score finale.** Dopo aver generato
   questi 3 numeri per un'entità, li combiniamo con il Teorema di Bayes
   per calcolare `P(minaccia | evidenze)` — la probabilità che
   quell'entità sia una minaccia, dato quello che sappiamo su di lei.
   Non è una media semplice: Bayes pesa ogni evidenza in base a quanto è
   *informativa* (una blacklist hit pesa diverso da una zona a rischio
   medio), partendo da una probabilità di base del 15% (assumiamo che
   il 15% delle entità nell'area sia effettivamente ostile).

**Perché "Non più poisoning random ma sui metadati":** invece di
alterare direttamente il Threat Score finale (un numero già calcolato),
l'attacco ora altera i dati di *input* (es. la blacklist di targhe nel
database) — più realistico, e l'errore si propaga naturalmente
attraverso tutta la catena di calcolo fino alla metrica finale (CEAE),
esattamente come farebbe un vero attacco a un database governativo.

### Slide 5 — Disaccoppiamento YOLO-Simulatore

YOLO è pesante da eseguire (frazioni di secondo per immagine). Il
simulatore tattico deve fare centinaia di step per ogni scenario, e
vorremmo poterlo rilanciare tante volte con condizioni casuali diverse
(più drone, più target) per avere risultati statisticamente solidi.
Farlo caricando YOLO ad ogni singolo step sarebbe troppo lento.
Soluzione: si misura **una volta sola, offline**, quanto YOLO
performa sotto attacco (F1-Score) su tutto il dataset VisDrone. Quel
numero (es. F1=0.720) diventa un parametro che il simulatore usa per
decidere probabilisticamente, ad ogni step, se il drone "vede" o no
un'entità — senza mai ricaricare la rete neurale. Nota sul Monte Carlo:
vedi il paragrafo di correzione in cima al documento.

### Slide 6 — Dataset per training

VisDrone fornisce split separati apposta: trainset (6471 immagini) per
allenare la patch, valset (531 immagini) per valutarla. Il motivo per
tenerli separati: se alleni e valuti sulle stesse immagini, la patch
"impara a memoria" quelle immagini specifiche e il numero di evasion
rate che misuri è artificialmente gonfiato — non riflette come si
comporterebbe su scene mai viste, che è quello che conta davvero per un
attacco "universale".

### Slide 7 — Divisore
Solo transizione, nessun contenuto da spiegare.

### Slide 8 — Fallback ibrido MPS
Un'operazione della libreria (`grid_sample`, usata per ruotare/scalare
la patch nell'EoT) non è implementata sul chip Apple M4 e mandava in
crash tutto, oppure costringeva a girare tutto su CPU (3 giorni
stimati). La soluzione (`PYTORCH_ENABLE_MPS_FALLBACK`) dice a PyTorch:
"fai girare *solo* questa operazione specifica su CPU, tutto il resto
resta sul chip veloce" — un compromesso mirato, non un abbandono
dell'accelerazione hardware.

### Slide 9 — Le 6 configurazioni a confronto (spiegazione completa riga per riga)

Questa è la slide più densa, vale la pena saperla a memoria nella
sequenza logica — ogni riga risponde a una domanda aperta dalla riga
precedente:

**Riga 0 — Baseline storico (Hinge Loss, 3000 update, nessun
accumulo).** Il primissimo esperimento, prima di ogni modifica recente.
Risultato: F1=0.760, evasion=38.7%. È il nostro punto di paragone
"prima".

**Riga 1 — Prima versione con Loss Asintotica (accum=4, 375 update
reali).** Abbiamo sostituito la Hinge con l'Asintotica per il problema
del gradiente morto (vedi glossario). Con accumulo 4 e budget di 1500
step raw, otteniamo solo 375 aggiornamenti reali dei pixel — molto
meno del baseline (3000). Risultato: F1=0.740, evasion=41.25%.
Leggermente meglio, ma non un salto netto. **Domanda aperta**: è perché
ci sono troppo pochi aggiornamenti reali?

**Riga 2 — Test con accumulo dimezzato (accum=2, 750 update reali,
stesso tempo di calcolo).** Per rispondere alla domanda della riga 1,
dimezziamo l'accumulo (4→2): a parità di step totali, questo raddoppia
gli aggiornamenti reali (375→750) senza costare più tempo. Risultato:
F1=0.750, evasion=40.0% — **non è migliorato**, anzi leggermente peggio.
**Nuova domanda**: forse il problema non è quanti aggiornamenti, ma
qualcos'altro nello scheduling?

**Riga 3 — Scoperta di un bug nello scheduler + budget molto più
grande (accum=4, 2500 update reali, scheduler corretto).** Qui abbiamo
scoperto che il Learning Rate (quanto "grandi" sono i passi
dell'ottimizzatore) non stava mai scendendo come dovrebbe per un bug di
configurazione — restava quasi al massimo per tutto il training invece
di calare gradualmente. Corretto il bug, e nello stesso momento abbiamo
alzato drasticamente il budget (10.000 step raw, accum=4 → 2500 update,
6.6 volte la riga 1). Risultato: F1=0.720, **evasion=43.75%, il
migliore finora**. Un miglioramento reale, ma modesto rispetto
all'aumento di calcolo (6.6x tempo per +2.5 punti) — rendimenti
decrescenti.

**Riga 4 — Test con accumulo estremo (accum=16, 625 update reali,
stesso tempo di calcolo della riga 3).** Ipotesi: se il segnale di ogni
singola immagine è rumoroso, forse mediare su *molte più* immagini per
ogni aggiornamento (16 invece di 4) dà una direzione più pulita anche
con meno aggiornamenti totali. Risultato: F1=0.740, evasion=41.25% —
**peggio della riga 3**, non meglio. Prova che il segnale condiviso tra
immagini diverse è debole per natura: mediare su più immagini non lo
rinforza abbastanza da compensare il numero minore di aggiornamenti.

**Riga 5 — Cambio della formula della loss: aggregazione Top-K (accum=4,
1250 update, metà budget della riga 3).** Invece di continuare a
giocare con quantità di aggiornamenti e accumulo, cambiamo cosa
guarda la loss: non più la media su tutte le celle (che include tanto
sfondo irrilevante), ma la media sulle 20 celle più confidenti — la
tecnica di Thys et al. (2019). Risultato: F1=0.720, **evasion=43.75%,
identico alla riga 3, ma con la metà del calcolo (1250 vs 2500
update)**. Questo è il risultato chiave: stesso tetto massimo,
raggiunto in modo molto più efficiente — la prova che il collo di
bottiglia non è (più) l'ottimizzatore, ma qualcosa di strutturale (vedi
slide 10-12).

### Slide 10 — Diagnosi
Riassume perché, guardando la norma del gradiente misurata direttamente
(non ipotizzata), il segnale è debole (0.0007-0.005) ma mai a zero
esatto — quindi non un bug di backprop, ma un problema di rumore tra
immagini diverse (vedi riga 4 sopra come prova sperimentale).

### Slide 11 — Fix Top-K
Il dettaglio tecnico della riga 5 sopra, con la citazione a Thys et al.
(2019) e il perché generalizziamo il loro "massimo" a un "top-20": con
più persone per frame (fino a 3), un massimo puro guarderebbe solo la
persona più confidente e ignorerebbe le altre.

### Slide 12 — Letteratura UAV-specific
Il punto: i paper "classici" sulle adversarial patch (Thys, Brown,
DPatch) lavorano a livello del suolo, su persone grandi e vicine.
Cercando nella letteratura specifica su droni/VisDrone, si scopre che
nessuno con risultati forti attacca i pedoni — tutti scelgono veicoli,
esplicitamente perché i pedoni da vista aerea sono troppo piccoli in
pixel. Il nostro tetto del 44% è coerente con questo, non un'anomalia.

### Slide 13 — Immagini prima/dopo
Le due immagini mostrano evasione completa: la persona rilevata da YOLO
(box verde, confidenza visibile) sparisce del tutto dopo l'applicazione
della patch. Prova visiva concreta che l'attacco funziona davvero, non
solo nei numeri aggregati.

### Slide 14 — Direzione futura
La sequenza: (1) verificare con più dati l'ipotesi sulla dimensione del
bersaglio (il campione attuale per bersagli grandi è troppo piccolo,
11 casi); (2) se confermata, allineare i dati allo scenario tattico già
definito nel simulatore (drone a bassa quota, non l'intero range
eterogeneo di VisDrone); (3) solo allora un budget di calcolo lungo
(20.000 step) con la loss top-K ha una motivazione scientifica solida.

### Slide 15 — Conclusioni
I 4 punti di forza da lasciare come impressione finale: metodo
sistematico, diagnosi quantitativa (non intuizioni), confronto con
letteratura specifica (non solo generica), framework multi-dominio che
va oltre il singolo attacco vision.
