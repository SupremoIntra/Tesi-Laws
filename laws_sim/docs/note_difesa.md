# Note per difesa e presentazione — accumulatore

Documento di lavoro. Ogni voce è già verificata contro il codice o contro
`thesis_notes.md`. Non aggiungere nulla che non sia `[VERIFICATO]`.

---

## A. La catena dell'attacco in una frase

> È stato costruito un attacco avversariale fisico di tipo *evasion*, in fase di
> inferenza, contro il canale ottico di un rilevatore di persone su piattaforma
> aerea. L'attaccante non tocca né la rete di comunicazione né i pesi del
> modello: il suo unico strumento è una superficie stampata sul corpo. È stato
> poi misurato quanto il degrado così ottenuto si propaghi lungo la catena
> decisionale di un sistema di targeting simulato.

Cinque parole da usare, in quest'ordine: **superficie di attacco → vettore →
evasion → inferenza → transfer**.

---

## B. Sotto il cofano, in cinque passaggi

Da usare come singola slide con cinque box.

1. **La patch nasce come numeri liberi, non come pixel.** Si ottimizza un
   tensore `Z` senza vincoli, e il pixel si ricava con una sigmoide. Motivo: se
   si tagliassero i valori fuori intervallo, i pixel al bordo riceverebbero
   gradiente zero e resterebbero congelati per sempre.
2. **La patch viene incollata sul petto, in proporzione al corpo.** 50% della
   larghezza × 40% dell'altezza del riquadro = 20% dell'area. In proporzione e
   non in pixel fissi, perché un indumento reale scala con la persona.
3. **Ogni fotogramma viene visto 16 volte, tutte diverse.** Rotazioni, scale,
   colori, rumore. Si ottimizza la *media* su queste 16, non il caso migliore.
   Senza questo, la patch funziona solo nell'inquadratura esatta con cui è nata.
4. **YOLO risponde con migliaia di celle; ne contano 20.** Si prendono solo le
   celle il cui centro cade dentro il riquadro della persona, e di quelle solo
   le 20 più confidenti. Le altre sono periferia rumorosa e diluirebbero il
   segnale.
5. **Il ciclo si chiude ogni 4 fotogrammi.** La memoria consente un fotogramma
   alla volta, quindi si accumulano 4 gradienti prima di aggiornare.

---

## C. Le domande scomode, con risposta pronta

### C1. "La loss si muove dell'1.5% ma l'evasione del 28%. Non è incoerente?"

No, perché non misurano la stessa cosa. Fra le due ci sono tre passaggi non
lineari:

- la loss agisce sulla **media** delle 20 celle migliori, ma il rilevamento
  sopravvive se **una sola** cella supera la soglia. Media contro massimo.
- la soglia di confidenza è un **gradino**. Se le predizioni sono addensate
  vicino alla soglia, uno spostamento piccolo della distribuzione fa
  attraversare la soglia a molte di esse.
- la loss è una media su 16 trasformazioni aggressive; la valutazione avviene su
  una sola inquadratura, più favorevole all'attaccante.

**Frase da dire:** "L'efficacia non viene da un crollo della confidenza, viene
dall'attraversamento della soglia decisionale da parte di predizioni che erano
già vicine ad essa."

### C2. "Perché K=20 se l'ottimo misurato è 10 su Okutama e 37 su VisDrone?"

Perché l'analisi di K misura il rilevatore **senza patch**: è una verifica di
plausibilità della scala, non un predittore dell'efficacia dell'attacco. K=20
sta fra i due ottimi. E il dato empirico conferma: su Okutama, dove K=20 è il
doppio dell'ottimo teorico, l'evasione ottenuta è la **più alta** (+0.280 contro
+0.250). `[VERIFICATO]`

**Il vero risultato non è il numero, è la divergenza fra i due criteri**: F1
premia la copertura totale dell'impronta, la media geometrica penalizza la
diluizione nello sfondo. Dentro un bersaglio, solo un nucleo ristretto di celle
porta evidenza forte.

### C3. "Il learning rate completa il suo decadimento?"

No. L'orizzonte è calcolato sugli aggiornamenti *attesi*, ma i fotogrammi senza
bersagli sopra i 60 px vengono scartati prima dell'aggiornamento. Nel run
riportato il passo ha percorso circa il 55% della curva. `[VERIFICATO]` per
strumentazione diretta.

**Perché non invalida i risultati:** sei configurazioni di ottimizzazione molto
diverse fra loro convergono in una banda di cinque punti. Il collo di bottiglia
non è il programma del passo. Ed è dichiarato fra le limitazioni.

### C4. "La patch che valuti è la migliore che hai trovato?"

No, è quella dell'ultimo aggiornamento. Le due differiscono di 0.0009 in media
mobile della loss, quindi in pratica coincidono, ma la formulazione corretta è
"al termine del budget". `[VERIFICATO]`

### C5. "Quanti campioni ha visto davvero l'ottimizzazione?"

Circa il 55–62% delle iterazioni lanciate. Gli altri fotogrammi non contengono
bersagli sopra la soglia dimensionale e hanno peso nullo per costruzione: è il
filtro tattico, una scelta di progetto, non una perdita accidentale.
`[VERIFICATO]` con contatori strumentati: su 200 iterazioni, 125 utili e 75
scartate per peso nullo, zero scartate per altre cause.

### C6. "Se il modello di minaccia è black-box, perché ottimizzi in white-box?"

Sono due piani diversi. L'attaccante è **black-box rispetto al sistema in
esercizio**: non lo interroga, non ne conosce i pesi. Ottimizza in **white-box
su un surrogato** della stessa famiglia architetturale. Il nome tecnico della
configurazione è **transfer attack**. È l'ipotesi più favorevole all'attaccante,
quindi i numeri sono un **limite superiore**, non una stima operativa. La
trasferibilità non è stata misurata in questo lavoro.

### C7. "Perché la specificità R2 non cambia mai?"

Perché per costruzione la patch non viene composta sui fotogrammi negativi: non
c'è nessun bersaglio valido su cui applicarla. R2 è quindi invariante per
disegno sperimentale, non per caso. Il fatto che risulti invariante su
**entrambi** i dataset lo conferma. `[VERIFICATO]`

### C8. "Perché hai sottocampionato Okutama a 527 fotogrammi?"

Perché è video a 30 fps: fotogrammi consecutivi differiscono di uno o due pixel
e non sono osservazioni indipendenti. Il bootstrap assume indipendenza. Con
14210 fotogrammi correlati gli intervalli di confidenza risultano cinque volte
troppo stretti — pseudo-replicazione. Con passo 27 restano 527 fotogrammi a
0.9 secondi di distanza. **Le stime puntuali non cambiano, cambia
l'incertezza**, ed è così che deve essere. `[VERIFICATO]`

---

## D. Numeri da sapere a memoria (solo questi)

| | VisDrone (n=531, 640px) | Okutama (n=527, 960px) |
|---|---|---|
| Evasion PRE → POST | 0.175 → 0.425 | 0.497 → 0.777 |
| **Δ Evasion [CI 95%]** | **+0.250** [+0.156, +0.346] | **+0.280** [+0.228, +0.332] |
| Specificità R2 | 0.969 invariante | 0.960 invariante |
| p-value metriche primarie | < 0.0001 | < 0.0001 |

Un solo messaggio da questa tabella: **l'effetto si replica su due domini
indipendenti con condizioni di ripresa diverse, e i due valori sono vicini.**
Per una tesi vale più di un numero alto su un dataset solo.

Norma del gradiente osservata: 7×10⁻⁴ – 3×10⁻². Soglia di troncamento: 1.0. Il
troncamento non è mai intervenuto. È l'evidenza del segnale debole.

---

## E. Cosa NON difendere (concedere subito, con precisione)

Concedere questi punti rapidamente e con numeri è più forte che difenderli male.

1. **Physical domain gap.** L'attacco è interamente digitale. La penalizzazione
   di variazione totale è condizione necessaria, non sufficiente, alla
   riproducibilità su stampa. Nessun test su tessuto reale.
2. **Trasferibilità non misurata.** Regime transfer dichiarato, non quantificato.
3. **Livelli 2 e 3 sono proof-of-concept.** Entità sintetiche, distribuzioni
   scelte in progetto. I loro risultati hanno statuto di conseguenze di un
   modello, non di osservazioni. Solo il Livello 1 produce misure sperimentali.
4. **Risoluzione 960 e non 1280** su Okutama: vincolo di memoria verificato, con
   un costo dichiarato in riquadri validi persi.
5. **Stratificazione IoU-matched** non eseguita.

---

## F. Struttura proposta per i 10 minuti

| Tempo | Contenuto |
|---|---|
| 0:00–1:30 | Problema e domanda di ricerca. Una slide, nessun numero |
| 1:30–3:00 | Architettura a tre livelli (figura) + statuto epistemico dei tre livelli |
| 3:00–5:00 | Sotto il cofano, i cinque passaggi della sezione B |
| 5:00–6:30 | **Before/after** su fotogramma reale, con box disegnate |
| 6:30–8:30 | Forest plot degli intervalli di confidenza, due dataset affiancati |
| 8:30–10:00 | Limitazioni (sezione E) e sviluppi futuri: il C.A.R.E. Kit |

Figure candidate già disponibili: `runs_comparison.png`,
`k_selection_plots.png`, `fig_candele_okutama_960_stride27.png`, output di
`generate_before_after_images.py`.

**Attribuzione obbligatoria in didascalia** per qualunque fotogramma di
Okutama-Action: licenza CC BY-NC-SA 3.0.

---

## G. Da verificare prima della difesa

- [ ] Discrepanza fra `vision_metrics.json` (R1 pre 0.5012, post 0.2200, R2
      0.9699) e `full_report_okutama_960_stride27.json` (0.5033 / 0.2233 /
      0.9604). Probabile differenza di sottoinsieme o di stride fra le due
      esecuzioni. **Va risolta prima di citare numeri in Cap. 5**: il dato da
      citare è quello del report decorrelato a n=527
- [ ] Ampiezza esatta dell'intervallo pre/post del danno collaterale su
      VisDrone, marcato `[RITIRATO]` perché non replica alle soglie 0.3 e 0.7
- [ ] Tempo di wall-clock di un run completo, per la slide sui vincoli di
      piattaforma



# DOMANDE BASTARDE CAP 5

## 5.1
"Il learning rate finale è 0.0047, non lo 0.001 atteso — è un bug nello scheduler?"→ No: verificato analiticamente. Con 
𝑇
𝑚
𝑎
𝑥
=
2000
T
max
=2000 e 
𝑡
=
1110
t=1110 la formula del coseno dà 
0.0047263
0.0047263, identico a 6 cifre al valore loggato. Il divario viene dal numero di aggiornamenti mancati (1110 su 2000), non da un errore nel programma di decadimento.

"Solo il 55.5% delle iterazioni produce un aggiornamento — non è uno spreco di calcolo?"
→ È conseguenza diretta di una scelta di design già giustificata in Cap. 4 (§4.3.2, filtro di rilevanza tattica): un fotogramma senza bersaglio ≥60px ha peso nullo. Confermato per strumentazione diretta: unica causa di scarto, nessun caso di maschera vuota o bbox mancante.

"La patch valutata è quella dell'ultimo aggiornamento, non la migliore — perché?"
→ Comportamento del codice (optimize_universal ritorna sempre l'ultima); differenza rispetto al checkpoint migliore è 0.0009 sulla media mobile della loss — trascurabile rispetto all'oscillazione del training, dichiarato come limitazione, non nascosto.

"VisDrone parte da R1 PRE = 0.825, Okutama da 0.503 — le due baseline sono comparabili?"
→ No, e non lo saranno mai per costruzione (dataset diversi). Il confronto valido è sul delta appaiato (+0.250 vs +0.280), non sui livelli assoluti — punto che tratteremo esplicitamente in §5.2.4 quando arriviamo alla discussione, ma tienilo già a mente: è la domanda più prevedibile che un commissario può fare guardando i due forest plot affiancati.

## CAP 5 in generale
Perché VisDrone e Okutama non sono comparabili in assoluto → confronta solo i delta appaiati (+25 vs +28pp), mai i livelli.
Δc̄ (-1.5pp) vs ΔEvasion (+28pp), rapporto 1:18 → non è incoerenza, sono 3 non-linearità dichiarate (§divario).
+27.0pp per-bersaglio ≈ +28.0pp per-fotogramma → coerenza interna, l'argomento più solido in tavola.
Danno collaterale: perché uno è "ritirato" e l'altro no → Bonferroni fallito su VisDrone (p=0.0136 contro soglia 0.0083 richiesta), pattern monotono su Okutama.
Layer 2/3: se te lo chiedono, di' subito "proof-of-concept, 4 difetti accertati, non quantificato" — non provare a difendere numeri che non useremo.