# HANDOFF — Tesi LAWS (Daniele Intra, Sicurezza Informatica, UNIMI)
Ultimo aggiornamento: sessione del 16 luglio 2026

## 1. Identità progetto
- Titolo: "Sistemi d'Arma Autonomi Letali: criticità nel trattamento dei dati sensibili e analisi sperimentale tra vulnerabilità algoritmiche e giuridiche"
- Relatore: Prof. Matteo Buffa (focus giuridico/filosofico/geopolitico)
- Correlatore: Prof. Massimo Walter Rivolta (focus tecnico/ML, autore delle Linee Guida LaTeX)
- Repo GitHub: `supremointra/Tesi-LAWS` (clonabile) — struttura: `/Latex` (main.tex, bibliografia.bib, tesi.sty, img/), `/YOLO`, `/laws_sim` (simulazione Python per Cap.3)
- **IMPORTANTE**: a inizio di ogni sessione di lavoro sul codice, chiedere a Daniele se la repo è aggiornata prima di clonarla/leggerla (poi `git fetch` + `git reset --hard origin/main` per allinearsi, non `git pull` semplice se ci sono modifiche locali residue).

## 2. Metodo di lavoro (non negoziabile)
- **Human-in-the-loop**: Claude non scrive interi sottocapitoli di sua iniziativa. Fornisce ossatura, bullet, impalcature LaTeX commentate (🎯 Obiettivo / 🗺️ Scaletta / 📚 Spunti / ❓ Domande Guida), revisiona bozze, propone modifiche mirate "riga X → cosa cambiare" invece di riscrivere intere sezioni.
- Eccezioni concordate: quando Daniele chiede esplicitamente un paragrafo tecnico completo, Claude scrive, marcando sempre `[SPUNTO INTEGRAZIONE: ...]` dove Daniele deve intervenire.
- Quando Daniele chiede una revisione ("verifica che sia corretta"), Claude corregge SOLO quello che è stato scritto, senza riscrivere lo stile o l'impianto argomentativo scelto da Daniele.
- **Tono**: clinico, accademico, mai colloquiale ("in sintesi", "come abbiamo visto" vietati).
- **Tipografia**: virgolette solo `` `...' `` (mai `"..."`), niente `---` per gli incisi (virgole/parentesi), `\footcite{}` esclusivo (mai `\cite{}` in linea), `È` mai `E'`, accenti gravi ovunque tranne perché/poiché/nonché, sottocapitoli mai sotto 1 pagina (accorpare con `\textbf{Caso X:}`), corsivo riservato a termini stranieri/titoli (mai su riferimenti interni tipo "Cap. 1" o su parole italiane per enfasi).
- Capitoli numerati in lettere ("Capitolo Primo", non "1") via `\renewcommand{\thechapter}`; `\thesection` ridefinito a parte — **non toccare questo meccanismo**.
- Bibliografia: motore `biblatex`+`biber`. Entry `.bib`: graffe non virgolette, acronimi protetti con doppie graffe.
- Regola citazioni: mai inventare fonti. Ogni `\footcite` deve corrispondere a una fonte reale verificata (paper accademico, testo di legge/regolamento diretto, o giornalismo primario per fatti di cronaca). Regolamenti UE e leggi nazionali si citano per estremi diretti nel testo/nota, senza necessità di entry `.bib`, salvo richiesta esplicita di Daniele (vedi es. `legge_132_2025`).
- **Attenzione ai sottocapitoli/casi già esistenti**: prima di proporre un nuovo blocco `\textbf{Il caso X:}`, verificare sempre se esiste già (es. errore commesso in sessione: creato un duplicato "Il caso Italiano" quando "Il caso Italia" esisteva già dal Cap.1 — poi corretto integrando nel blocco esistente).

## 3. Stato reale dei capitoli (verificato via `git diff` sull'ultimo push, non solo dichiarato)
- **Cap.1**: completo e revisionato.
  - Non-paper Crosetto **integrato** (confermato) nel blocco `Il caso Italia:` già esistente (non un nuovo sottocapitolo), con paragrafo che collega l'approccio ``two way'' di Bencini alla dottrina italiana sulla minaccia ibrida (`crosetto_2025_nonpaper`).
  - Paragrafo Chat Control 1.0/2.0 in 2.1: confermato integrato.
- **Cap.2**:
  - Capoverso introduttivo: scritto (Loss Function, hidden layers, XAI, Automation Bias).
  - **2.1** (Necroetica/Tanatopolitica): completa.
  - **2.2** (``Black Box'' militare e MHC): **completata, revisionata a fondo e verificata riga per riga in questa sessione (nessun errore residuo)**. Contenuto: framing GRC del MHC, AI Act artt.14/15/22 + limite strutturale art. 2 par.3, Legge 132/2025 (che replica la stessa logica di esenzione, non se ne discosta — attenzione, prima bozza di Claude conteneva un'affermazione errata su questo punto, poi corretta), genesi negoziale dell'esenzione (trilogo, Francia, tensione con giurisprudenza CGUE), confronto DoD Directive 3000.09/HRW vs modello UE, posizione cinese (criteri cumulativi CCW GGE), XAI e Automation Bias, chiusura sul doppio cortocircuito giuridico/tecnico.
  - **2.3** (Bias e Dati: il caso Lavender e Habsora): **non ancora scritta**. Esiste già un'impalcatura dettagliata commentata nel file (obiettivo, scaletta in 5 punti, spunti bibliografici, domande guida) — vedi `main.tex` subito dopo la fine di 2.2. Daniele ha menzionato di avere appunti aggiuntivi presi su Notion da integrare, non ancora condivisi con Claude in nessuna sessione di questo progetto (verificato con `conversation_search`, nessun riscontro). **Prossima sessione: chiedere a Daniele di incollare/caricare questi appunti prima di procedere alla bozza.**
  - **2.4**: confermato completo con tutti e tre gli inserimenti tecnico-dialogici (Bai/CAI, Sleeper Agents, Shevlane/Structured Access) e la correzione "all lawful purposes".
  - **2.5**: ancora impalcatura vuota. Attenzione: le fonti "SHADE-Arena" e "Agents of Chaos" menzionate da Daniele per questa sezione non sono mai state verificate con una fonte pubblica precisa — non inserire `\footcite` finché non le fornisce lui.
  - Bug storico del `\section{Il collasso...}` duplicato: risolto, confermato (una sola occorrenza).
- **Cap.3, Cap.4**: solo scaletta/appunti grezzi (esiste `/laws_sim` con codice Python funzionante per gli esperimenti del Cap.3).

## 4. Bibliografia — chiavi aggiunte nelle sessioni recenti (confermate presenti e corrette in `bibliografia.bib`)
`crosetto_2025_nonpaper`, `dod_directive_3000_09`, `hrw_2023_dod3009review`, `mako_2026_china_mhc`, `legge_132_2025`, `vogiatzoglou_2024_ai_act_exception`.

Fonte accademica trovata ma **non ancora aggiunta** per metadati incompleti (verificare autore/editore/anno prima di usarla): capitolo "La sicurezza e la difesa nazionale nella legge [132/2025]" (Pignanelli, cognome confermato dall'URL, resto da verificare), su art. 6 Legge 132/2025 — utile se si vuole approfondire ulteriormente il carve-out sicurezza/difesa italiano.

## 5. Pulizia bibliografia (lavoro di rifinitura, NON durante la scrittura delle sottosezioni)
8 entry in `bibliografia.bib` risultano non citate da nessuna parte (nemmeno negli spunti commentati per le sezioni non ancora scritte), verificate con `grep` incrociato tra chiavi definite e `\footcite`/`\cite` nel documento:
`cools_2024`, `erendor_2024`, `guo_2025`, `koch_2022`, `oliveri_2024`, `raska_2023`, `stanley_lockman_2021`, `stanley_lockman_2023`.
La maggior parte sembra pertinente a 2.2/2.3 (già scritta o in arrivo) e a Cap.4: da rivalutare una per una a fine stesura, non rimuovere a cuor leggero. Solo `erendor_2024` e `raska_2023` sembrano davvero generiche/orfane.
Inoltre `cina_2023_poisoning` è citata solo nello spunto commentato per 2.3 (Wild Patterns Reloaded, Cinà/Grosse/Demontis 2023) — non è "morta", semplicemente non ancora usata nel testo attivo.

## 6. Reminder comportamentali attivi (già in memoria Claude, non serve ripeterli)
- Ricordare di refreshare la repo GitHub quando si lavora sul codice.
- A fine sessione, includere sempre nell'HANDOFF il reminder di pulire le entry `.bib` non citate (vedi sez. 5), da fare a fine stesura.
- Quando Daniele dice "mando la mail a Buffa" o "mi ha risposto il relatore" → ricordargli la possibilità di esportare i capitoli in `.docx` via pandoc per il ciclo di revisione con Buffa (che non usa LaTeX).

## 7. Prossimi passi suggeriti (in ordine)
1. Ottenere da Daniele gli appunti Notion su Lavender/Habsora e scrivere la bozza di 2.3, seguendo l'impalcatura già presente nel file.
2. Scrivere 2.5 (Agent Chaos) — verificare prima le fonti SHADE-Arena/Agents of Chaos con Daniele.
3. Fondere Cap.3 con gli output reali di `/laws_sim`.
4. A fine stesura completa: rivalutare le 8 entry `.bib` non citate (sez. 5) per eventuale rimozione.