# Note per la difesa — v3

Ricostruito dal `note_difesa.md` precedente, **verificato riga per riga contro
`main.tex`** (Capitoli 4 e 5) e contro `thesis_notes.md`.

Convenzioni:

- `[V]` = numero o affermazione ritrovata nel testo della tesi. Se non c'è `[V]`,
  è un ragionamento, non una citazione.
- I riferimenti sono **per nome di sezione**, non per numero di sotto-sezione:
  Capitoli 4 e 5 non hanno sotto-sezioni, quindi «§4.3.2» non esiste e citarlo in
  sede di difesa è un autogol gratuito.
- Numerazione utile: Cap. 4 → 4.1 architettura, 4.2 modello di minaccia, 4.3
  funzione obiettivo, 4.4 metriche, 4.5 insiemi di dati, 4.6 protocollo
  statistico, 4.7 disegno degli esperimenti. Cap. 5 → 5.1 ottimizzazione, 5.2
  efficacia, 5.3 discussione.

**Modifiche rispetto alla versione precedente.** Rimossi i riferimenti a §5.2.4 e
§4.3.2 (inesistenti). Riscritta la risposta su Bonferroni, che era sbagliata nel
merito. Sostituito «danno collaterale» con «rilevamenti spuri» ovunque indichi la
metrica. Chiusa la voce aperta sulla discrepanza `vision_metrics.json`, risolta
in `thesis_notes.md` §8.7. Aggiunte otto domande nuove, tutte su punti che il
testo attuale espone e che la versione precedente non copriva.

---

# 0. Tre riflessi da automatizzare

Prima di ogni contenuto. Sono i tre punti dove si perde una difesa altrimenti
solida, e vanno risposti senza pensarci.

1. **Qualunque domanda che confronti i due domini si risponde sul delta
   appaiato**, mai sui livelli assoluti. `+0.250` contro `+0.280`. Le linee di
   base non sono confrontabili e non lo saranno mai: sono insiemi di dati diversi.
2. **$R_2$ invariante non significa «attacco pulito».** Significa che nei
   fotogrammi su cui $R_2$ è calcolata la patch non è mai stata composta. Vedi
   B4: la risposta istintiva è quella sbagliata.
3. **Se una domanda contiene un numero, ripeterlo prima di rispondere.** Serve a
   guadagnare due secondi e a non scambiare `531` con `527`.

`n = 531` → **VisDrone**, immagini statiche, 640 px, nessun sottocampionamento.
`n = 527` → **Okutama-Action**, video, 960 px, passo 27. La vicinanza è voluta:
il passo 27 è stato scelto per ottenere numerosità comparabili.

---

# 1. Scheda numerica

## 1.1 Da sapere alla cifra (solo questi)

| | VisDrone ($n=531$, 640 px) | Okutama ($n=527$, 960 px) |
|---|---|---|
| Evasion rate PRE → POST | $0.175 \to 0.425$ | $0.497 \to 0.777$ |
| **Δ Evasion [IC 95%]** | **$+0.250$** [$+0.156$, $+0.346$] | **$+0.280$** [$+0.228$, $+0.332$] |
| $R_2$ | $0.969$, invariante | $0.960$, invariante |
| $p$ metriche primarie | $< 0.0001$ | $< 0.0001$ |

Un solo messaggio: **l'effetto si replica su due domini indipendenti con
condizioni di ripresa diverse, e i due valori sono vicini.** `[V]` Tab. 5.3 e 5.5.

## 1.2 Da saper ricostruire (non memorizzare, derivare)

`[V]` tutti dalle tabelle del Capitolo 5.

- $R_1$: $0.825 \to 0.575$ e $0.503 \to 0.223$. È $1 -$ evasion, non è un numero
  in più da imparare.
- $\sqrt{R_1R_2}$: $\Delta = -0.148$ e $-0.232$. $F_1$: $\Delta = -0.168$ e $-0.300$.
- Fotogrammi positivi: $15.1\%$ e $57.0\%$ → **80 e 300 fotogrammi**. Serve per B3.
- Bersagli validi per fotogramma: $0.72$ e $1.45$.
- Tasso di falso allarme di base: $3.1\%$ e $4.0\%$, cioè $1 - R_2$.
- Budget: 8000 iterazioni, 2000 aggiornamenti attesi, **1110 eseguiti (55.5%)**.
- Learning rate finale $0.004726$ contro $0.001$ atteso.
- Norma del gradiente $6.8\times10^{-4}$ – $3.0\times10^{-2}$ su Okutama,
  $0.0007$ – $0.005$ sulle sei configurazioni VisDrone. Troncamento a $1.0$ **mai
  raggiunto in alcun run**.
- Stratificazione Okutama, bucket $[60,100)$ px: $70.0\% \to 97.0\%$, **$+27.0$ pp
  su $n = 19921$ bersagli**; bucket $[100,150)$: $+2.9$ pp su $n = 618$;
  bucket $[150,\infty)$: vuoto.
- $\Delta\bar{c} = -1.5$ pp ($0.5465 \to 0.5312$) contro $\Delta$Evasion
  $= +28.0$ pp. Rapporto $\approx 1{:}18$.
- $K$ ottimo: $244/37$ su VisDrone, $\sim55/\sim10$ su Okutama. Adottato $K=20$.
- Evasion PRE su VisDrone: $3.75\%$ a soglia $0.30$, $72.5\%$ a soglia $0.70$.
- Sei configurazioni: evasion fra $38.7\%$ e $43.75\%$.

## 1.3 Verifiche di coerenza interna — l'arma migliore che hai

Sono controlli che ho rifatto sui numeri della tesi e che **tornano**. Se ti
mettono in difficoltà su un dato, uno di questi te ne fa uscire.

- I rilevamenti spuri su VisDrone sono tutti multipli di $1/80$:
  $0.4625 = 37/80$, $0.1125 = 9/80$, $0.0500 = 4/80$. Su Okutama tutti multipli
  di $1/300$: $0.3700 = 111/300$, $0.2867 = 86/300$, $0.0967 = 29/300$,
  $0.0433 = 13/300$. Coerente con $15.1\%$ di 531 e $57.0\%$ di 527.
- I bersagli della stratificazione, $19921 + 618 = 20539$, divisi per i 14210
  fotogrammi del valset completo danno $1.445$, cioè il $1.45$ dichiarato.
- $19921/20539 = 96.99\%$, cioè il «$97\%$ dei bersagli validi» del testo.
- $n=300$ e $n=8095$ della nota su Bonferroni sono il $57.0\%$ di 527 e di 14210.
- Coseno con $T_{max}=2000$, $t=1110$ → $0.0047263$, contro $0.004726$ loggato.

---

# 2. L'impianto, da dire a memoria

## 2.1 In una frase

> È stato costruito un attacco avversariale di tipo *evasion*, in fase di
> inferenza, contro il canale ottico di un rilevatore di persone su piattaforma
> aerea. L'attaccante non tocca né la rete di comunicazione né i pesi del
> modello: il suo unico strumento è una superficie applicata sul corpo. È stato
> poi misurato quanto il degrado così ottenuto si propaghi lungo la catena
> decisionale di un sistema di targeting simulato.

Cinque parole, in quest'ordine: **superficie di attacco → vettore → evasion →
inferenza → transfer**.

## 2.2 Sotto il cofano, cinque passaggi

Una slide, cinque riquadri.

1. **La patch nasce come numeri liberi, non come pixel.** Si ottimizza un tensore
   senza vincoli e il pixel si ricava con una sigmoide. Se si tagliassero i
   valori fuori intervallo, i pixel al bordo riceverebbero gradiente zero e
   resterebbero congelati.
2. **Si applica sul petto, in proporzione al corpo.** $50\%$ della larghezza
   $\times$ $40\%$ dell'altezza del riquadro, cioè il $20\%$ dell'area. In
   proporzione e non in pixel fissi, perché un indumento reale scala con la
   persona.
3. **Ogni fotogramma viene visto 16 volte, tutte diverse.** Rotazioni, scale,
   colori, rumore. Si ottimizza la *media* su queste 16, non il caso migliore.
4. **YOLO risponde con migliaia di celle; ne contano 20.** Solo quelle il cui
   centro cade dentro il riquadro della persona, e di quelle solo le 20 più
   confidenti.
5. **Il ciclo si chiude ogni 4 fotogrammi.** La memoria consente un fotogramma
   alla volta, quindi si accumulano 4 gradienti prima di aggiornare.

---

# 3. Domande, per categoria

## A. Metodo e ottimizzazione

### A1. «Il learning rate finale è $0.0047$, non lo $0.001$ atteso. È un bug nello scheduler?»

No, ed è verificato analiticamente. Con $T_{max}=2000$ e $t=1110$ la formula del
coseno dà $0.0047263$, identico a sei cifre al valore registrato `[V]` Tab. 5.1.
Il divario viene dal numero di aggiornamenti mancati, non dal programma di
decadimento. Lo scheduler è corretto; è il contatore degli aggiornamenti a
fermarsi prima.

### A2. «Solo il $55.5\%$ delle iterazioni produce un aggiornamento. Non è calcolo sprecato?»

È conseguenza diretta del **filtro di rilevanza tattica** dichiarato nella
Sezione 4.3: un fotogramma privo di bersagli di altezza $\geq 60$ px ha peso
nullo e non contribuisce. `[V]` La strumentazione diretta su un campione di 200
iterazioni, con contatori su tutti i punti di uscita, attribuisce **75 scarti su
200** a quel filtro, coerente con il $55.5\%$ del run completo. Nessun caso di
maschera vuota o riquadro mancante. Non è spreco: è il costo di una scelta di
disegno dichiarata a priori.

*Se insistono:* la conseguenza è che il budget effettivo è 1110 aggiornamenti,
non 2000, e da questo discende A1. I due fatti sono lo stesso fatto.

### A3. «La patch valutata è quella dell'ultimo aggiornamento, non la migliore. Perché?»

Perché è il comportamento della funzione di ottimizzazione, che restituisce
sempre l'ultimo stato. `[V]` Lo scarto rispetto al checkpoint di minima media
mobile è $0.0009$, contro una media mobile che vale $0.7583$: quattro ordini di
grandezza sotto. È dichiarato in tabella, non nascosto. `[V]` La media mobile è
piatta dall'aggiornamento 840 in poi, quindi «ultimo» e «migliore» sono lo stesso
punto entro il rumore.

### A4. «Perché $K=20$ se gli ottimi misurati sono $\sim10$ su Okutama e $37$ su VisDrone?»

`[V]` Perché $K=20$ è compreso fra i due ottimi secondo il criterio primario, la
media geometrica $\sqrt{R_1R_2}$, e le curve di selezione sono calcolate **senza
patch** sugli insiemi di validazione: non è un iperparametro sintonizzato
sull'esito dell'attacco.

**Concedere onestamente:** la giustificazione «è in mezzo ai due» è formulata a
posteriori, perché la curva di Okutama è stata misurata dopo. Quello che si può
sostenere è che $K$ non è stato scelto guardando l'evasion rate.

*Perché non $F_1$ come criterio:* `[V]` l'ottimo secondo $F_1$ è di ordine
centinaia su VisDrone e decine su Okutama, mentre secondo la media geometrica
resta di ordine decine su entrambi. La divergenza misura l'estensione del nucleo
di celle a evidenza forte rispetto all'impronta geometrica completa del
bersaglio.

### A5. «Perché ottimizzare in white-box se il modello di minaccia è black-box?»

Regime *transfer* con corrispondenza esatta fra surrogato e bersaglio: stessa
architettura, stessi pesi. I risultati sono quindi un **limite superiore**, non
una stima operativa. `[V]` La trasferibilità verso un rilevatore diverso non è
stata misurata ed è dichiarata fra le limitazioni. Dirlo prima che lo dicano
loro.

### A6. «Il fattore di accumulo 4 non è troppo basso? Con più batch andrebbe meglio.»

`[V]` È stato provato: la configurazione 4 usa accumulo 16 contro il 4 della
configurazione 3, e **non produce miglioramento** ($41.25\%$ contro $43.75\%$ di
evasion). Questo è un argomento a favore, non contro: se il vincolo fosse il
rumore del gradiente, l'aumento dell'accumulo lo avrebbe ridotto. Il vincolo è
altrove, nel segnale disponibile.

### A7. «Perché una copertura del $20\%$ e non maggiore?»

Vincolo di realizzabilità come capo di abbigliamento, ripreso dalla letteratura
sulle camuffature fisiche universali. Aumentarla renderebbe il numero più bello e
il dispositivo meno indossabile. Su bersaglio di 60–100 px di altezza, quel
$20\%$ è una superficie di poche decine di pixel per lato: è il vero limite
fisico dell'esperimento.

### A8. «La patch si vede a occhio nudo. Che mimetismo è?»

Non è mimetismo percettivo, è una perturbazione ottimizzata contro un rilevatore.
La sua visibilità all'occhio umano è **irrilevante rispetto all'obiettivo
dichiarato**, che è la mancata rilevazione da parte del modello. Il vincolo di
banda spaziale che ne determina l'aspetto a bassa frequenza è per di più ciò che
la rende riproducibile a stampa.

## B. Statistica — il terreno del correlatore

### B4 va letta prima di tutte le altre.

### B1. «La correzione di Bonferroni: come l'hai applicata?»

`[V]` Uniformemente, su sei confronti — tre soglie per due insiemi di dati —
quindi con soglia richiesta $p < 0.0083$.

- **VisDrone**, soglia $0.5$: $p = 0.0136 > 0.0083$. **Il risultato è ritirato.**
- **Okutama**, soglia $0.5$: $p < 0.0001$. **Sopravvive.**
- **Okutama**, soglia $0.3$: $p = 0.0108$. Significativo al livello convenzionale
  $0.05$, **non** dopo correzione. Dichiarato come tale.
- Soglia $0.7$ su entrambi: nessun evento, $p = 1.000$.

La regola è la stessa per tutti e sei. Non c'è un caso trattato con due pesi.

### B2. «Perché sei confronti? Due sono degeneri, con zero eventi.»

Domanda giusta, e ho la risposta. Se si contassero solo i quattro confronti
informativi, la soglia salirebbe a $0.05/4 = 0.0125$. **La conclusione su
VisDrone non cambia**: $0.0136$ resta sopra $0.0125$, il risultato resta ritirato.
Cambierebbe una sola casella, Okutama a soglia $0.3$, che passerebbe da non
significativa a significativa per un margine di $0.0017$. La scelta di $m=6$ è
quindi quella conservativa, e l'unica conclusione che ne dipende è già riportata
come debole.

### B3. «Perché su VisDrone i rilevamenti spuri non reggono e su Okutama sì?»

Prima la ragione formale — Bonferroni, vedi B1 — e poi quella sostanziale, che è
più convincente: **la metrica su VisDrone poggia su 80 fotogrammi positivi**,
il $15.1\%$ di 531. I conteggi sono $9/80$ contro $4/80$. Su Okutama i fotogrammi
positivi sono 300 e i conteggi $29/300$ contro $13/300$. Non è un dominio più
docile, è un campione quasi quattro volte più grande sulla quantità in esame.

`[V]` Il segno è negativo su entrambi i domini: la perturbazione **riduce** i
rilevamenti privi di corrispondenza.

### B4. «Perché la specificità $R_2$ non cambia mai? Fino alla sedicesima cifra.»

**Non rispondere «perché l'attacco è pulito».** È la lettura naturale ed è
sbagliata.

`[V]` La causa è strutturale, non sperimentale. La patch è composta
esclusivamente sui riquadri dei bersagli validi. Nei fotogrammi **privi** di
bersaglio valido — che sono esattamente quelli su cui $R_2$ è calcolata —
l'immagine sottoposta al rilevatore è **identica bit per bit** nelle due
condizioni, e l'inferenza di YOLOv8 è deterministica. Stesso ingresso, stessa
uscita. Non c'è nulla da misurare.

Tre conseguenze da avere pronte:

- $R_2$ misura il **tasso di falso allarme di base del rilevatore sullo sfondo**,
  $3.1\%$ su VisDrone e $4.0\%$ su Okutama. È una proprietà del dataset e del
  modello, non dell'attacco.
- Ne discende che **l'intero movimento di $\sqrt{R_1R_2}$ è attribuibile a
  $R_1$**. Nel disegno di questo esperimento la media geometrica è una
  trasformazione monotona di $R_1$ e non aggiunge informazione. Concederlo.
- **$R_2$ non misura i falsi positivi indotti dalla patch.** Quelli sono i
  *rilevamenti spuri*, che sono una metrica diversa, calcolata sui fotogrammi
  positivi, e che infatti si muovono. Se qualcuno confonde le due, questa è la
  correzione.

`[V]` L'invarianza non è un artefatto del punto di taglio: alle soglie $0.3$,
$0.5$ e $0.7$ su VisDrone, $R_2$ vale $0.8248$, $0.9690$ e $1.0000$, e in tutti e
tre i casi è identica fra pre e post. E la replica su un secondo dominio esclude
che sia una proprietà accidentale del primo.

### B5. «Hai sottocampionato Okutama da 14210 a 527 fotogrammi. Non hai buttato via potenza statistica?»

Non c'era potenza da buttare. È video a 30 fps: fotogrammi consecutivi
differiscono di uno o due pixel e non sono osservazioni indipendenti, mentre il
bootstrap assume indipendenza. `[V]` Il confronto è in tabella: la stima puntuale
passa da $+0.2812$ a $+0.2800$, cioè coincide entro il terzo decimale, mentre
l'ampiezza dell'intervallo aumenta di un fattore prossimo a cinque. **Cambia la
stima dell'incertezza, non la conclusione:** $p < 0.0001$ in entrambi i casi.

I 14210 fotogrammi non erano 14210 osservazioni. Il passo 27 restituisce 527
fotogrammi distanti $0.9$ secondi. `[V]` Il guadagno del secondo insieme di dati
è nel dominio osservato, non nella numerosità.

### B6. «La stratificazione per taglia è su $n = 19921$ bersagli. Ma quel campione è correlato.»

**Questa è la domanda più tecnica che puoi ricevere, e la risposta esiste.**

Sì: la stratificazione è condotta sull'insieme completo, 14210 fotogrammi
($19921 + 618 = 20539$ bersagli, cioè $1.45$ per fotogramma). Ma **non viene
riportato alcun intervallo di confidenza né alcun $p$** per quei $+27.0$ pp: è
una stima puntuale, usata per localizzare l'effetto lungo la scala del bersaglio,
non per un'inferenza.

E la legittimità del confronto con i $+28.0$ pp del campione decorrelato è
dimostrata proprio dalla Tabella sulla decorrelazione: **le stime puntuali sono
invarianti** al sottocampionamento, cambia solo l'incertezza. Confronto due
stime puntuali, e non rivendico incertezza su nessuna delle due.

### B7. «Nella nota sui rilevamenti spuri citi $p = 0.0038$ su $n = 8095$, che è il campione correlato. Non contraddice la Sezione 4.6?»

**Punto debole reale. Concedere subito e con precisione.**

Quel valore è riportato come controprova di **direzione**, non di significatività,
in un caso in cui il campione decorrelato non contiene alcun evento e la misura
non è quindi calcolabile. Il $p$ su campione correlato è affetto da
pseudo-replicazione ed è sistematicamente troppo piccolo: non può fondare una
conclusione, e infatti nessuna conclusione della tesi vi si appoggia.

*Nota per Daniele, non per la commissione:* valuta se togliere il valore numerico
dalla didascalia e lasciare solo «conferma la stessa direzione». Il numero non
serve e apre un fianco.

### B8. «Nessuna replica dell'ottimizzazione con seed diversi. Come sostieni che il risultato non sia rumore?»

**Non esiste difesa diretta. Concedere in due secondi, poi passare all'indiziaria.**

«È corretto, non ci sono repliche con seed diversi, ed è dichiarato fra le
limitazioni.» Poi, senza pausa, le due prove indiziarie:

1. `[V]` **Sei configurazioni** su VisDrone, che variano formulazione della loss,
   criterio di aggregazione, fattore di accumulo e programma di decadimento,
   convergono in una banda di cinque punti percentuali, fra $38.7\%$ e $43.75\%$.
   È variabilità su assi più grossolani del seed.
2. `[V]` **Replica su un secondo dominio indipendente**, con risoluzione,
   piattaforma di ripresa e distribuzione di scala diverse, che restituisce
   $+0.280$ contro $+0.250$.

Nessuna delle due sostituisce le repliche con seed. Entrambe rendono
improbabile che il risultato sia un artefatto di una singola inizializzazione.

### B9. «Il bootstrap: perché appaiato, e perché 10 000 iterazioni?»

Appaiato perché ogni fotogramma è osservato in **entrambe** le condizioni: la
variabilità fra fotogrammi è enormemente maggiore della differenza fra
condizioni, e trattare i due campioni come indipendenti la sprecherebbe. È lo
stesso motivo per cui gli intervalli sul delta sono più stretti di quanto
suggerirebbero gli intervalli sui livelli.

## C. Interpretazione dei risultati

### C1. «La loss si muove dell'$1.5\%$ e l'evasione del $28\%$. Non è incoerente?»

No, perché non misurano la stessa cosa. `[V]` $\bar{c}$ passa da $0.5465$ a
$0.5312$, e fra quel delta e il delta di evasione ci sono **tre non linearità
formalizzate nella Sezione 4.3**:

- la loss agisce sulla **media** delle 20 celle migliori, ma il rilevamento
  sopravvive se **una sola** cella supera la soglia. Media contro massimo.
- la soglia di confidenza è un **gradino**. Se le predizioni sono addensate
  attorno alla soglia, uno spostamento piccolo della distribuzione ne fa
  attraversare la soglia a molte.
- la loss è una media su 16 trasformazioni aggressive; la valutazione avviene su
  una singola inquadratura, più favorevole all'attaccante.

**Frase da dire:** l'efficacia non viene da un crollo della confidenza, viene dal
fatto che molte predizioni erano già vicine alla soglia. Il rapporto $1{:}18$ è
l'ordine di grandezza atteso da quelle tre trasformazioni, non una sorpresa.

### C2. «Su Okutama l'attacco è più efficace.»

**Trappola. Cambia la linea di base, non l'efficacia.**

`[V]` Il contributo marginale è $+25$ e $+28$ punti percentuali: quasi identico.
Ciò che cambia fra i due insiemi è la condizione di partenza del rilevatore, non
l'entità dell'effetto attribuibile all'attacco. In assenza di attacco, Okutama
parte da un evasion rate del $49.7\%$ contro il $17.5\%$ di VisDrone.

*Se chiedono perché la baseline è peggiore:* `[V]` non è stato determinato. Le
ipotesi sono uno scarto di dominio del rilevatore rispetto alle posture riprese
dall'alto e la distorsione da ridimensionamento a canvas quadrato; **nessuna
delle due è stata verificata**, e la verifica — stratificazione per taglia della
sola condizione di controllo — è fra gli sviluppi futuri. Non inventare una
spiegazione: è scritto nella tesi che non si sa.

### C3. «$+27.0$ pp per bersaglio e $+28.0$ pp per fotogramma. Coincidenza?»

**È l'argomento più solido che hai in mano. Usalo tu, prima che lo chiedano.**

Due unità di analisi diverse — il bersaglio contro il fotogramma — e due criteri
di attribuzione distinti, misurati indipendentemente, convergono entro un punto
percentuale. Non è una coincidenza numerica: è una verifica di coerenza interna
fra due misure che avrebbero potuto divergere e non lo fanno.

### C4. «Perché l'effetto è tutto nel bucket $[60,100)$ px?»

`[V]` Perché quel bucket contiene il $97\%$ dei bersagli validi. Il bucket
$[100,150)$ è in **saturazione**: la condizione di controllo è già al $97.1\%$ e
non resta margine misurabile, tanto che il delta è $+2.9$ pp su 618 casi. Il
bucket $[150,\infty)$ è vuoto per costruzione, perché a 960 px una soglia di 150
px equivale a 200 px alla risoluzione nativa di 1280.

Il **regime dei bersagli grandi resta non testato**, su entrambi i domini. È una
limitazione dichiarata.

### C5. «Perché la perturbazione *riduce* i rilevamenti spuri? Non è controintuitivo?»

`[V]` Il segno negativo indica una perturbazione che sopprime il rilevamento del
soggetto su cui è applicata **senza alterare il comportamento del rilevatore sul
resto della scena**. Se la patch generasse rumore diffuso, il segno sarebbe
opposto. Che sia l'unica quantità su cui il secondo dominio produce un risultato
nuovo anziché una replica è dichiarato nel testo.

### C6. «Il limite osservato è del tuo metodo o del compito?»

`[V]` Del compito, e ci sono tre evidenze convergenti. La norma del gradiente,
fra $0.0007$ e $0.005$, con troncamento a $1.0$ **mai** intervenuto in alcun run:
il segnale è debole, non instabile. L'aumento del fattore di accumulo da 4 a 16
che non produce guadagno: il budget non è il vincolo. La media mobile piatta
dall'aggiornamento 840 su 1110. `[V]` La letteratura concorda: i lavori con tassi
di successo elevati su VisDrone bersagliano **veicoli**; l'ottimizzazione
multi-dimensionale delle caratteristiche **scarta esplicitamente la classe
pedone** per l'esiguità in pixel da vista zenitale; gli attacchi efficaci su
immagini aeree operano su bersagli più grandi e sulle rappresentazioni intermedie
anziché sull'uscita finale. La formulazione top-$K$ è ripresa da attacchi a
livello del suolo, dove il soggetto occupa **due ordini di grandezza** in più
dell'immagine.

## D. Portata, limiti, livelli 2 e 3

### D1. «I livelli 2 e 3 producono numeri. Perché non li presenti come risultati?»

`[V]` Perché hanno quattro difetti accertati per lettura diretta del codice:

1. l'esito del rilevamento non è un test statistico ma un'estrazione da
   distribuzione uniforme con soglia fissa;
2. la condizione di riferimento non legge le metriche del Livello 1 in assenza di
   attacco, e confronta quindi una costante di letteratura con il dato misurato
   sotto attacco;
3. la specificità non entra nella catena decisionale per le entità civili;
4. l'effetto della distanza è conteggiato due volte, nella metrica empirica e nel
   modello analitico, in contrasto con la Sezione 4.1.

`[V]` L'unico enunciato sostenibile è che il sistema a canali fusi mostra un
degrado inferiore rispetto al canale visivo isolato, coerente con la funzione di
ridondanza attribuita alla fusione. **L'entità del fenomeno non è quantificabile
con i dati disponibili.** La correzione è definita nel suo perimetro e non è
stata eseguita.

Questa asimmetria è **dichiarata in apertura del Capitolo 5**, non scoperta sotto
esame. Dirlo.

### D2. «E la metrica CEAE?»

Non è fra i risultati. Dipende da conteggi del simulatore, ed è irrecuperabile
combinando i dati della visione con i due livelli successivi: richiede conteggi a
livello di **decisione di ingaggio**, mentre il Livello 1 produce conteggi a
livello di **fotogramma di detection**. Sono unità di analisi diverse. È fra gli
sviluppi futuri.

### D3. «Il C.A.R.E. Kit esiste?»

No, ed è dichiarato come sviluppo futuro. `[V]` La perturbazione caratterizzata
in questa tesi ne è la **componente percettiva**; l'obiettivo 4 è formulato di
conseguenza, come «misurazione dei limiti di una contromisura passiva».

### D4. «Perché 960 px e non 1280?»

Vincolo di memoria, verificato, con un costo dichiarato in riquadri validi persi.
È anche la ragione per cui il bucket $[150,\infty)$ è vuoto.

### D5. «Perché YOLOv8n e non un modello più grande?»

**Verificato: la motivazione non è scritta da nessuna parte.** «YOLOv8 nano»
compare una volta sola in tutta la tesi, fra parentesi, nella Sezione 4.1. Non
c'è un paragrafo che giustifichi la scelta della versione. Se te lo chiedono,
non fingere che ci sia.

Risposta onesta, in due tempi. Primo, la ragione sostanziale: la versione nano è
rappresentativa dell'inferenza a bordo su piattaforma con vincoli energetici, che
è lo scenario del lavoro, ed è anche l'unica compatibile con il vincolo di
memoria che ha già imposto la risoluzione di 960 px. Secondo, la concessione: un
modello più capace sposterebbe la **linea di base** — probabilmente verso l'alto,
quindi con più margine di degrado disponibile — ma non cambierebbe il modello di
minaccia né la natura del limite, che la Sezione 5.3 attribuisce alla dimensione
in pixel del bersaglio e non alla capacità della rete.

Quello che **non** puoi dire è che l'attacco funzionerebbe altrettanto bene su
una versione più grande: non è stato misurato, ed è la stessa lacuna della
trasferibilità (A5).

## E. Il ponte con il Capitolo 2 — la zona più esposta

### E1. «Quanto sostiene davvero il dato sul piano giuridico?»

**Il rischio qui è duplice: gonfiare o svalutare. Serve la formula esatta.**

Quello che il dato sostiene: `[V]` che su una rete di *Computer Vision* come YOLO
**la soglia di confidenza non è un dettaglio implementativo ma il parametro che
decide chi viene visto**. Sul medesimo insieme di dati e in assenza di qualunque
attacco, la frazione di persone non rilevate passa da meno del quattro percento a
soglia $0.3$ a oltre il settanta percento a soglia $0.7$. Una singola variabile
iperparametrica, priva di contenuto semantico, ridefinisce per intero la
popolazione che il sistema considera presente nella scena. **Questo è misurato,
non argomentato.**

Quello che il dato **non** sostiene: nessuna conclusione su un sistema d'arma
reale, la cui architettura non è nota. LAWS-SIM `[V]` non è un sistema d'arma né
una replica funzionale.

**Frase di chiusura:** «La risposta che ne emerge non è una contromisura pronta
all'uso, ma una misura della fragilità del presupposto di fatto su cui l'intera
delega si regge.» È la formulazione della tesi, ed è quella giusta anche a voce.

### E2. «Non è un salto, dal Capitolo 2 al Capitolo 5?»

Il Capitolo 2 argomenta che il vuoto non è soltanto normativo ma anche tecnico,
perché l'opacità della *Black Box* rende inverificabile a posteriori la catena di
attribuzione che il diritto presuppone. Il Capitolo 5 misura una cosa sola e
circoscritta: quanto sia manipolabile il canale percettivo su cui il principio di
distinzione viene verificato di fatto. Il ponte è quello, e non di più.

### E3. «"Danno collaterale" nel Capitolo 2 e "rilevamenti spuri" nel Capitolo 5 sono la stessa cosa?»

No, ed è il motivo per cui i nomi sono diversi. Nel Capitolo 2 «danno
collaterale» ha il significato del diritto internazionale umanitario. Nei
capitoli tecnici la quantità misurata è il conteggio di rilevamenti privi di
corrispondenza con un bersaglio annotato: un evento del rilevatore, non un
evento sul terreno. La distinzione terminologica è deliberata.

## F. Fuori perimetro — riconoscerle e non entrarci

Domande su opportunità politica dei LAWS, posizione dell'Italia, moratoria,
responsabilità penale del comandante. Sono legittime e sono terreno del relatore
titolare, non del dato sperimentale.

Formula: «Il Capitolo 2 tratta questo aspetto sul piano [normativo / etico]; il
contributo sperimentale non aggiunge evidenza su quel punto specifico.» Poi, se
c'è, la posizione argomentata del capitolo. Non trascinare i numeri dove non
arrivano.

## G. «Descrivimi l'architettura» — la domanda che sembra facile

Il relatore l'ha già posta una volta. Sembra un invito a recitare uno schema a
blocchi, e la risposta istintiva — «l'immagine entra, YOLO rileva, il simulatore
decide» — **è sbagliata**. Chi ha visto il codice se ne accorge subito. La
domanda non chiede i blocchi: chiede se sai cosa passa nella giuntura fra di
essi.

### G1. La frase di apertura

> Non è una pipeline. Sono un livello sperimentale e un livello simulato,
> deliberatamente **disaccoppiati**, e fra i due passa un file JSON che contiene
> due numeri.

**Il simulatore non contiene YOLO.** Non carica i pesi, non vede un pixel, non
esegue inferenza. Contiene un'estrazione di Bernoulli il cui parametro è stato
misurato su YOLO da un'altra parte.

Dirlo per primo risolve la domanda in una frase e mostra che la giuntura la
conosci. Tutto il resto è approfondimento su richiesta.

### G2. Perché disaccoppiati — due argomenti, in quest'ordine

`[V]` Il primo è di costo: valutare YOLOv8 a ogni passo temporale imporrebbe un
costo proibitivo, dato il prodotto fra numero di entità visibili e centinaia di
iterazioni per scenario. L'inferenza è quindi eseguita una volta sola, in
modalità non interattiva, sull'intero insieme di validazione.

`[V]` Il secondo è quello buono, e va detto per secondo perché trasforma un
vincolo in una scelta di disegno: **il disaccoppiamento rende i due livelli
verificabili indipendentemente**. Il comportamento del sensore si valida sui dati
reali isolandolo dalla simulazione; il simulatore si collauda iniettando
parametri arbitrari senza richiedere nuove inferenze. Il JSON è l'interfaccia
esclusiva, e rende il trasferimento di informazione integralmente ispezionabile.

### G3. Cosa attraversa davvero il ponte

Passano $R_1$ e $R_2$, misurate a livello di **fotogramma**, nelle varianti pre e
post attacco. Nient'altro. `[V]` A ogni passo temporale, per ciascuna entità
entro il raggio operativo, l'esito del rilevamento è un'estrazione di Bernoulli:

| Entità | Contromisura | Parametro dell'estrazione |
|---|---|---|
| ostile | attiva | $R_1$ **post** attacco |
| ostile | inattiva | $R_1$ **pre** attacco |
| civile (caso negativo) | — | $1 - R_2$, tasso di falso allarme |

**La conseguenza da dire tu, prima che la trovino loro:** `[V]` poiché la patch
non è mai composta sui fotogrammi privi di bersaglio, $R_2$ è invariante e **il
ramo civile collassa su un unico valore**. La separazione delle due varianti
resta nell'architettura per compatibilità con scenari futuri in cui la
contromisura sia applicata anche a soggetti non ostili, non perché produca due
comportamenti distinti oggi. È il terzo dei quattro difetti dichiarati (D1).

Formulazione brutale e corretta, da tenere in tasca: **l'attacco non si propaga
nel simulatore come attacco. Si propaga come sostituzione di un numero.**

### G4. «Perché $R_1$ e non $F_1$ come parametro del sensore?»

Ragione probabilistica, non estetica. `[V]` $F_1$ è una media armonica calcolata
sull'intero insieme di validazione e **non ammette interpretazione probabilistica
riferita al singolo fotogramma**, mentre l'ambiente simulato valuta a ogni
iterazione l'evento binario del rilevamento di una specifica entità. Serve un
numero che sia il parametro di una Bernoulli, e $F_1$ non lo è. È anche il motivo
per cui $F_1$ resta metrica **secondaria** in tutto il Capitolo 5.

### G5. «E la distanza fra sensore e bersaglio?»

`[V]` Calcolata a tre dimensioni e registrata nei log, ma **non altera l'esito
dell'estrazione**. Deliberato: $R_1$ è una media empirica su fotogrammi reali e
incorpora già nativamente una distribuzione eterogenea di distanze e di
dimensioni apparenti. Un ulteriore fattore di decadimento conteggerebbe lo stesso
effetto due volte.

**Attenzione, qui c'è la trappola.** La Sezione 4.1 stabilisce il principio, ma
il codice del simulatore lo viola: `[V]` il doppio conteggio è il **quarto
difetto dichiarato** nella Sezione 5.3. Se la domanda arriva, la risposta è
«il principio è quello, e nel simulatore non è rispettato — è uno dei quattro
difetti per cui il Livello 2 non produce risultati quantitativi». Concedere,
non difendere.

### G6. Il Livello 3, in trenta secondi

`[V]` Il canale visivo confluisce in una fusione bayesiana con due canali
indipendenti, OSINT e analisi comportamentale, con pesi $0.45 / 0.35 / 0.20$ e
probabilità a priori $0.50$. Il `DecisionAgent` applica tre soglie — $0.58$
ingaggio, $0.38$ allerta, $0.22$ tracciamento — più due vincoli precauzionali che
declassano l'ingaggio ad allerta: ampiezza dell'intervallo di confidenza mobile
oltre $0.40$ su finestra delle dieci osservazioni più recenti, oppure più di tre
civili prossimi con punteggio sotto $0.85$.

La fusione è il percorso lungo cui il degrado percettivo si propaga, **attenuato**
dalla validazione incrociata. E l'unico enunciato sostenibile — canali fusi meno
degradati del canale visivo isolato — non è quantificabile con i dati
disponibili (D1).

### G7. Il versante vision, in ordine causale

Non elencare i componenti. **Derivali dai due vincoli del modello di minaccia**,
perché è così che il Capitolo 4 li costruisce e perché mostra che nessuno di essi
è arbitrario.

`[V]` **Modello di minaccia.** Superficie di attacco limitata al canale di
acquisizione ottica. Avversario privo di accesso all'infrastruttura logica del
velivolo e ai pesi del rilevatore in esercizio. Unico vettore ammesso: una
superficie stampata applicata sul corpo. È un attacco di *evasion* in fase di
inferenza — pesi e corpus di addestramento restano integri, la deviazione è
indotta per via del solo input.

`[V]` **Primo vincolo: localizzazione.** L'avversario controlla la sola porzione
di scena occupata dal proprio corpo. Da qui due conseguenze.

- *Parametrizzazione sigmoide.* Il troncamento soddisfa il vincolo di dominio ma
  congela: un elemento saturato al bordo riceve gradiente nullo per il resto
  dell'ottimizzazione e genera una regione inerte. Il cambio di variabile
  logistico di Carlini e Wagner fa sì che il vincolo sia soddisfatto **per
  costruzione anziché per proiezione**, il gradiente resti non nullo ovunque e
  nessun elemento diventi inerte.
- *Collocazione proporzionale*, $50\%$ della larghezza per $40\%$ dell'altezza del
  riquadro, centrata sul terzo superiore del tronco. Proporzionale e non fissa in
  pixel perché un indumento sottende una frazione approssimativamente invariante
  della figura di chi lo indossa: una regione fissa sarebbe un capo di taglia
  variabile e, in prospettiva aerea, eccederebbe l'impronta del soggetto
  coprendogli il volto.

`[V]` **Secondo vincolo: realizzabilità fisica.** La perturbazione deve
conservare efficacia attraverso la catena stampa–tessuto–ottica–campionamento,
sotto illuminazione e geometria variabili. Da qui le **16 trasformazioni EoT** —
si ottimizza la media su di esse, non il caso migliore — e la **penalizzazione di
variazione totale**, che vincola la banda spaziale della superficie. È
quest'ultima a determinare l'aspetto morbido e a bassa frequenza della patch, ed
è condizione **necessaria ma non sufficiente** alla riproducibilità a stampa.

**Funzione obiettivo.** Si prendono le celle il cui centro cade dentro il
riquadro del bersaglio, di quelle le $K = 20$ più confidenti, e se ne minimizza
la media in forma asintotica. Il **filtro di rilevanza tattica** azzera il peso
dei fotogrammi privi di bersagli sopra i 60 px — da cui i 1110 aggiornamenti su
2000 (A2). L'**accumulo su 4 fotogrammi** è vincolo di memoria: il batch fisico è
uno.

### G8. La nota che vale la domanda — da verificare nel codice

La loss opera sulle **celle** della testa di YOLO, cioè sull'uscita grezza prima
della soppressione dei non massimi. La valutazione opera sui **rilevamenti**,
dopo NMS, a soglia $0.50$ e IoU $0.3$. Se il codice lo conferma, ottimizzazione e
valutazione guardano due viste diverse della stessa rete, e questa è una
**quarta** ragione strutturale del divario $1{:}18$ fra $\Delta\bar{c}$ e
$\Delta$Evasion, che oggi la tesi spiega con tre (C1).

**Non usarla finché non hai controllato `patch_optimizer.py`.** Se confermata è
una risposta che nessun commissario si aspetta; se non lo è, è un'invenzione.

---

# 4. Cosa concedere subito, con precisione

Concedere questi punti in dieci secondi, con il numero, è più forte che
difenderli male. Sono già tutti scritti nel Capitolo 6.

1. **Divario col dominio fisico.** L'attacco è interamente digitale. La
   penalizzazione di variazione totale è condizione necessaria, non sufficiente,
   alla riproducibilità su stampa. Nessun test su tessuto reale.
2. **Trasferibilità non misurata.** Regime transfer dichiarato, non quantificato.
3. **Nessuna replica con seed diversi.**
4. **Livelli 2 e 3 proof-of-concept**, quattro difetti accertati.
5. **Baseline più debole su Okutama, causa non determinata.**
6. **Risoluzione 960 e non 1280**, vincolo di memoria.
7. **Regime dei bersagli grandi non testato** su entrambi i domini.
8. **La decorrelazione temporale non aumenta la potenza statistica.**
9. **$\sqrt{R_1R_2}$ non aggiunge informazione a $R_1$ in questo disegno**,
   perché $R_2$ è invariante per costruzione.

---

# 5. Piano di studio, tre categorie

**Alla cifra (una decina di numeri).** La tabella 1.1 per intero. Il $55.5\%$ e i
1110 aggiornamenti. Il $+27.0$ pp su $n=19921$. Il $-1.5$ contro $+28.0$. Le
soglie $0.0083$ e $0.0136$.

**Da ricostruire, non memorizzare.** Tutto il paragrafo 1.2. Le tre non
linearità di C1. I quattro difetti del simulatore. Le tre evidenze di C6. Se le
sai derivare, non serve impararle.

**Da concedere.** La sezione 4 per intero, più B7 e la formulazione a posteriori
di A4.

**Ordine di ripasso**, per pericolosità decrescente: **G1–G3** (già chiesta una
volta e già andata male: ha la precedenza su tutto), poi E1, B8, B4, C2, quindi
B6 e B7, infine il resto.

---

# 6. Ancora aperto

- [ ] Pulizia dei campi `note` in `bibliografia.bib` — verificare quali delle
      sette sono davvero appunti di lavoro. La voce [42] è priva di autore.
- [ ] Valutare la rimozione del valore $p = 0.0038$ dalla didascalia sui
      rilevamenti spuri (vedi B7).
- [x] ~~Verificare che la motivazione di YOLOv8n sia scritta~~ — **non lo è**.
      «YOLOv8 nano» compare una sola volta, fra parentesi, nella Sezione 4.1.
      Risposta orale predisposta in D5.
- [ ] Verificare in `patch_optimizer.py` se la loss agisce prima della
      soppressione dei non massimi mentre la valutazione agisce dopo (vedi G8).
      Se confermato, va aggiunto a C1 come quarta non linearità.
- [ ] Tempo di wall-clock di un run completo, per la slide sui vincoli di
      piattaforma. Non presente in `thesis_notes.md`.
- [ ] Decidere se inserire la figura prima/dopo anche in tesi, oltre che nelle
      slide.
- [x] ~~Discrepanza `vision_metrics.json` / `full_report_okutama`~~ — **risolta**,
      `thesis_notes.md` §8.7: stessa misura su campionamenti diversi,
      `vision_metrics.json` è il valset completo senza passo. Il dato da citare
      resta quello del report decorrelato a $n=527$.
