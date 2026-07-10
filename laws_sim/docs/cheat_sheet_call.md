# Cheat sheet — da leggere a voce, niente di scontato

Formato: **Termine** → cos'è → a cosa serve / perché l'ho usato.

---

## Glossario lampo

- **Hinge Loss** → penalità che diventa zero appena la confidenza scende
  sotto una soglia fissa → l'abbiamo abbandonata perché sotto soglia il
  gradiente muore (zero esatto), il training si blocca.
- **Loss Asintotica** (`-log(1-conf)`) → penalità che non è mai
  esattamente zero, per nessun valore di confidenza → sempre un minimo
  di spinta a migliorare, niente gradiente morto.
- **Evasion Rate** → % di frame con una persona vera in cui YOLO, sotto
  attacco, non rileva nessuno → il numero principale per giudicare
  l'attacco.
- **Accumulo (gradient accumulation)** → sommare i suggerimenti di N
  immagini prima di aggiornare davvero la patch → simula un batch più
  grande senza usare più memoria; N alto = meno aggiornamenti reali a
  parità di immagini processate.
- **EoT** → media della loss su 16 versioni ruotate/scalate/colorate
  della patch → la costringe a funzionare in condizioni reali diverse,
  non su un'immagine esatta.
- **TV Loss** → penalizza pixel adiacenti troppo diversi tra loro →
  peso alto forza macchie di colore ampie, che sopravvivono
  all'interpolazione e sono stampabili fisicamente.
- **Top-K** → la loss guarda solo le 20 celle più confidenti, non tutte
  → concentra il segnale dove conta, invece di diluirlo sullo sfondo.
- **Norma del gradiente / clipping** → quanto è "forte" il segnale di
  aggiornamento → mai scattato il limite di sicurezza in nessun run,
  segno che il problema è un segnale debole, non uno esploso.
- **F1-Score** → media di Precision (quando dice "persona" ha ragione?)
  e Recall (quante persone vere trova?) → sotto attacco la Precision
  resta 1.000 sempre, tutto il calo viene dal Recall.

---

## OSINT — la domanda "si contrappone alla patch?"

No: sono due sensori indipendenti. La patch inganna la **vista**
(YOLO). L'OSINT è un secondo canale — metadati (targa, geo-rischio,
tracce social), non pixel — pensato per riflettere sistemi reali di
data fusion (Palantir e simili), non una singola telecamera. Un
attivista può ingannare la telecamera ma restare identificato dai
metadati: per questo il framework modella entrambi i canali, non solo
la vista.

- **Bernoulli** (blacklist targa) → evento sì/no → 75% target, 1% civili.
- **Beta** (geo-rischio) → numero continuo 0-1 → Beta(8,2) target
  (piegata verso alto rischio), Beta(2,8) civili.
- **Poisson** (tracce social) → conteggio di eventi discreti → media 12
  per target, media 1 per civili.
- **Bayes** → combina questi 3 numeri in un Threat Score finale, pesato
  per quanto ogni evidenza è informativa, partendo da un 15% di
  probabilità di base.

---

## Le 6 configurazioni — una riga ciascuna, in sequenza logica

0. **Baseline (Hinge, 3000 update)** → F1 0.760, evasion 38.7% → punto
   di partenza, prima di ogni modifica.
1. **Asintotica, accum=4, 375 update** → 41.25% → meglio ma poco.
   *Domanda: troppo pochi aggiornamenti?*
2. **Asintotica, accum=2, 750 update** (doppio aggiornamento, stesso
   tempo) → 40.0%, **non migliora**. *Domanda: allora cos'altro?*
3. **Bug scheduler scoperto e corretto + budget 6.6x (2500 update)** →
   **43.75%, il migliore** → miglioramento vero ma con rendimenti
   decrescenti (tanto calcolo per poco guadagno).
4. **Accumulo estremo, 16, 625 update, stesso tempo della riga 3** →
   41.25%, **peggio** → prova che mediare su più immagini non basta, il
   segnale condiviso tra scene è debole per natura.
5. **Cambio la loss: Top-K invece di media (Thys et al. 2019), 1250
   update, metà budget della riga 3** → **43.75%, identico alla riga 3
   con metà calcolo** → stesso tetto, raggiunto meglio: prova che il
   limite non è più l'ottimizzatore, è strutturale (dimensione dei
   bersagli, vedi letteratura UAV).

---

## Se il run notturno finisce in tempo

**Pre-flight check reale (trainset VisDrone), eseguito prima del training:**

```
Annotazioni persona totali:        86.958
>= 60px (soglia minima usata sempre): 1.556  (1.8%)
>= 80px (tatticamente rilevanti):       394  (0.5%)
```

**Il 98.2% di tutte le annotazioni è sotto la soglia minima utilizzabile.**
Questo è un dato più forte di qualunque cosa avessimo prima (la
stratificazione sul valset era su un campione già ridotto) — è la prova
quantitativa diretta, sulla maggioranza del dataset, che il soffitto è
domain shift, non ottimizzatore. **Vale come slide anche da solo, a
prescindere dal risultato del training.**

Con solo 394 annotazioni ≥80px sparse su 5684 frame, la maggior parte
degli 8000 step notturni salta la loss (peso zero) — atteso, non un
bug. Se il salto di evasion rate sarà modesto, non è un fallimento: è
un'ulteriore conferma (scarsità di dati tattici, non solo dimensione).

Di' esattamente questo, con qualunque numero esca dall'eval:
*"Il pre-flight mostra che il 98.2% del dataset è sotto soglia
utilizzabile — la maggioranza, non un'eccezione. Questo conferma con un
dato quantitativo il domain shift. Il training di stanotte pesa la loss
di conseguenza; se il segnale tattico disponibile è troppo scarso per
convergere in una notte, il passo naturale è un dataset image-specific
(Piano B, Wu et al. 2020) che non soffre di questa scarsità."*

Non dire un numero di evasion rate prima di averlo visto stampato.
