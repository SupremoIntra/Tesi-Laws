# HANDOFF — Tesi LAWS (Daniele Intra, Sicurezza Informatica, UNIMI)
Ultimo aggiornamento: sessione del 12 luglio 2026 (giorno prima: 11 luglio)

## 1. Identità progetto
- Titolo: "Sistemi d'Arma Autonomi Letali: criticità nel trattamento dei dati sensibili e analisi sperimentale tra vulnerabilità algoritmiche e giuridiche"
- Relatore: Prof. Matteo Buffa (focus giuridico/filosofico/geopolitico)
- Correlatore: Prof. Massimo Walter Rivolta (focus tecnico/ML, autore delle Linee Guida LaTeX)
- Repo GitHub: `supremointra/Tesi-LAWS` (pubblica, clonabile) — struttura: `/Latex` (main.tex, bibliografia.bib, tesi.sty, img/), `/YOLO`, `/laws_sim` (simulazione Python per Cap.3)
- **IMPORTANTE**: a inizio di ogni sessione di lavoro sul codice, chiedere a Daniele se la repo è aggiornata prima di clonarla/leggerla.

## 2. Metodo di lavoro (non negoziabile)
- **Human-in-the-loop**: Claude non scrive interi sottocapitoli di sua iniziativa. Fornisce ossatura, bullet, impalcature LaTeX commentate (🎯 Obiettivo / 🗺️ Scaletta / 📚 Spunti / ❓ Domande Guida), revisiona bozze, propone modifiche mirate "riga X → cosa cambiare" invece di riscrivere intere sezioni.
- Eccezioni concordate: quando Daniele chiede esplicitamente un paragrafo tecnico completo (es. sintesi di un paper con spazio per integrare), Claude scrive, marcando sempre `[SPUNTO INTEGRAZIONE: ...]` dove Daniele deve intervenire.
- **Tono**: clinico, accademico, mai colloquiale ("in sintesi", "come abbiamo visto" vietati).
- **Tipografia**: virgolette solo `` `...' `` (mai `"..."`), niente `---` per gli incisi (virgole/parentesi), `\footcite{}` esclusivo (mai `\cite{}` in linea), `È` mai `E'`, accenti gravi ovunque tranne perché/poiché/nonché, sottocapitoli mai sotto 1 pagina (accorpare con `\textbf{Caso X:}`).
- Capitoli numerati in lettere ("Capitolo Primo", non "1") via `\renewcommand{\thechapter}`; `\thesection` ridefinito a parte per restare numerico (1.1, 1.2) — **non toccare questo meccanismo**, è stato debuggato con attenzione (rischio: senza il secondo renewcommand i paragrafi diventano "Primo.1").
- Bibliografia: motore `biblatex`+`biber` (non più bibtex/cite). Entry `.bib`: graffe non virgolette, acronimi protetti con doppie graffe, campi minimi per tipo. Niente URL con `%20` o caratteri encoded dentro `\url{}` annidato — usare `url={...}` diretto (bug riscontrato e corretto in sessione precedente).
- Regola citazioni: mai inventare fonti. Ogni `\footcite` deve corrispondere a una fonte reale verificata (paper accademico o, per fatti di cronaca 2026, giornalismo primario). Se non verificabile in sessione, segnalarlo esplicitamente come "da verificare" invece di fabbricare.

## 3. Stato reale dei capitoli (al netto di quanto pushato su GitHub)
- **Cap.1**: completo, revisionato, in fase di refinement minore (accenti, corsivi, coerenza già corretti in sessioni precedenti).
- **Cap.2**:
  - Capoverso introduttivo: scritto, denso, ben integrato tecnicamente (Loss Function, hidden layers, XAI, Automation Bias).
  - **2.1** (Necroetica/Tanatopolitica): bozza avanzata, in revisione continua. Contiene: genealogia Foucault→Mbembe→Buffa, esempio Metalhead, paragrafo tecnico IoBT (da `kufakunesu_2025_iobt`), chiusura Eschaton/soglia statistica. Ultima modifica proposta ma **non confermata da Daniele**: aggiunta paragrafo sul parallelo Chat Control 1.0/2.0 (UE, luglio 2026) come esempio di Dual-Use dell'infrastruttura di raccolta dati in ambito civile — verificare se è stato integrato.
  - **2.2, 2.3**: ancora impalcatura vuota (`[SCRIVI QUI...]`), non ancora scritte da Daniele.
  - **2.4**: testo scritto e già pubblicato, ma **non ancora aggiornato** con il layer tecnico-dialogico concordato (Constitutional AI/Bai 2022, Sleeper Agents/Hubinger 2024, Structured Access/Shevlane 2022, correzione "all lawful uses"→"all lawful purposes", aggiunta paragrafo sul seguito legale/injunction). Vedi sez. 5 sotto per i blocchi di testo pronti da inserire.
  - **2.5**: ancora impalcatura vuota.
  - **Bug noto da correggere**: `\section{Il collasso dell'autoregolamentazione...}` duplicato (due occorrenze, una vuota da eliminare) — verificare se già risolto.
- **Cap.3, Cap.4**: solo scaletta/appunti grezzi, non ancora sviluppati narrativamente (esiste però `/laws_sim` con codice Python funzionante per gli esperimenti).

## 4. Fatti reali 2026 verificati in sessione (non serve riverificarli, salvo notizie più recenti)
- **Operation Absolute Resolve** (gennaio 2026, raid USA-Venezuela/Maduro) e uso di Claude via Palantir: reale, riportato da Axios, WSJ, Reuters, CNN.
- **Enciclica "Magnifica Humanitas"** di Papa Leone XIV (15 maggio 2026): reale, testo su vatican.va.
- **Disputa Anthropic–Pentagono**: designazione "supply chain risk" (marzo 2026), causa legale, injunction della giudice Rita Lin che la definisce ritorsione pretestuosa, corte d'appello che nega la sospensione. Reale, ben documentato (Axios, CNN Business, Reuters, voce Wikipedia dedicata).
- **Chat Control UE**: il 9 luglio 2026 il Parlamento UE ha prorogato "Chat Control 1.0" (deroga ePrivacy, scansione volontaria CSAM, esclusa crittografia E2E) fino ad aprile 2028; negoziato in corso su "Chat Control 2.0" (permanente, potenzialmente client-side scanning). Reale, fonti: Euronews, Il Post, ANSA, The Record.
- **NON verificate/da non usare senza controllo**: "SHADE-Arena" e "Agents of Chaos" (citati nel knowledge base iniziale di Daniele per la 2.5) — non ho trovato fonte pubblica precisa in nessuna sessione. Non inserire `\footcite` per questi finché Daniele non fornisce il riferimento esatto.

## 5. Blocchi di testo pronti per l'integrazione nella 2.4 (non ancora confermato se applicati)
Tre inserimenti puntuali concordati (vedi sessione precedente per il testo integrale):
1. Dopo la frase su Constitution/`carter_2026`: frase su Constitutional AI/RLAIF come meccanismo tecnico (`\footcite{bai_2022_cai}`).
2. Correzione lessicale "all lawful uses"→"all lawful purposes" + frase su Sleeper Agents (`\footcite{hubinger_2024_sleeper}`) sul paradosso "l'accusa del DoD ha fondamento tecnico reale".
3. Frase finale allungata su Palantir/Structured Access (`\footcite{shevlane_2022_access}`), versione estesa già fornita.

## 6. Bibliografia — chiavi aggiunte in queste sessioni (verificare presenza in bibliografia.bib)
`horowitz_scharre_2015`, `kallenborn_2021`, `sharkey_2016`, `wood_2024_xai`, `horowitz_kahn_2024`, `kufakunesu_2025_iobt`, `cina_2023_poisoning`, `bai_2022_cai`, `shevlane_2022_access`, `hubinger_2024_sleeper`, `state_resp_2025` (corretta con autore reale Demir, Şamil), `euronews_chatcontrol_2026`, `ilpost_chatcontrol_2026` (queste ultime due proposte, verificare se aggiunte).

## 7. Reminder comportamentali attivi (già in memoria Claude, non serve ripeterli)
- Ricordare di refreshare la repo GitHub quando si lavora sul codice.
- Quando Daniele dice "mando la mail a Buffa e avremo finito anche il capitolo 2" o "mi ha risposto il relatore" → ricordargli la possibilità di esportare i capitoli in `.docx` via pandoc per il ciclo di revisione con Buffa (che non usa LaTeX).

## 8. Prossimi passi suggeriti (in ordine)
1. Verificare se le modifiche proposte per 2.4 e il paragrafo Chat Control in 2.1 sono state integrate su GitHub.
2. Continuare stesura 2.2 (Black Box/MHC) e 2.3 (Lavender/Habsora) — impalcature già pronte nelle sessioni precedenti.
3. Fondere Cap.3 con gli output reali di `/laws_sim`.
4. Integrare il non-paper Crosetto nel Cap.1 (ancora in sospeso da sessioni precedenti).
