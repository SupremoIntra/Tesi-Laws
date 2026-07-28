# -*- coding: utf-8 -*-
"""
GRAFICO "A CANDELA" (punto + baffi) -- versione 527 frame (full_report.json)
Revisione per pubblicazione: registro formale, nessuna sovrapposizione,
forest plot del pannello (b) privo di annotazioni numeriche (i valori vivono
nella Tabella associata).

Convenzioni grafiche adottate
-----------------------------
- Pannello (a): livelli assoluti pre e post. Rappresentazione descrittiva.
  In un disegno appaiato la sovrapposizione visiva degli IC pre/post NON e'
  prova di assenza di effetto; il giudizio di significativita' si fa sul (b).
- Pannello (b): forest plot della differenza appaiata Delta = post - pre, con
  asse x simmetrico rispetto all'ipotesi di nullita' (Delta = 0). Nessun valore
  numerico e' riportato nel tracciato: direzione, ampiezza e incertezza si
  leggono dalla geometria, mentre stima puntuale, IC e p-value sono demandati
  alla Tabella. Un IC che non interseca la linea di nullita' corrisponde a un
  effetto significativo al livello considerato.
- Le barre di errore codificano l'intervallo di confidenza bootstrap al 95%
  (B repliche); il campo "mean" del JSON e' usato solo come controllo di
  centratura e non viene disegnato.
- Specificita' R2 e danno collaterale medio per frame hanno Delta = 0 con
  IC = [0, 0] (candela degenere): non sono tracciati e sono discussi nel testo.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# =============================================================================
# 1. PERCORSI (risoluzione robusta, indipendente dalla working directory)
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def resolve_report_path(name="full_report.json"):
    """Restituisce il primo percorso esistente tra una lista di candidati;
    altrimenti solleva FileNotFoundError elencando i tentativi effettuati."""
    candidates = [
        PROJECT_ROOT / "outputs" / "metrics" / name,
        SCRIPT_DIR   / name,
        Path.cwd()   / "outputs" / "metrics" / name,
        Path.cwd()   / name,
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = "\n    - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Report '{name}' non trovato. Tentativi:\n    - {tried}"
    )

PATH_REPORT = os.environ.get("REPORT_PATH") or str(resolve_report_path())

# =============================================================================
# 2. LETTURA E NORMALIZZAZIONE DELLE CHIAVI
# =============================================================================
def strip_keys(o):
    """Rimuove spazi iniziali/finali da tutte le chiavi (ricorsivo)."""
    if isinstance(o, dict):
        return {k.strip(): strip_keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_keys(x) for x in o]
    return o

with open(PATH_REPORT, "r", encoding="utf-8") as f:
    FR = strip_keys(json.load(f))

METRICS = FR["metrics"]
N_ITER  = FR["n_iter"]     # repliche bootstrap
N_FRAME = FR["n_frame"]    # frame analizzati

# =============================================================================
# 3. SELEZIONE E ORDINAMENTO DELLE METRICHE
# =============================================================================
# Ordine dall'alto verso il basso: outcome composito, F1, componenti di R1.
ORDINE = [
    "sqrt(R1*R2)",
    "F1 (secondaria)",
    "Sensitivita' R1",
    "Evasion rate (1-R1)",
]
DEGENERI = ["Specificita' R2", "Collateral (media/frame)"]

def is_informative(m):
    """Vero se l'IC del Delta ha ampiezza non nulla (baffi effettivamente
    presenti). Filtro di sicurezza contro candele degeneri."""
    d = m["paired_delta"]
    return (d["high"] - d["low"]) > 1e-9

names = [n for n in ORDINE if n in METRICS and is_informative(METRICS[n])]

# Etichette dell'asse y: un solo spazio di separazione, nessuna sigla interna.
LABEL = {
    "sqrt(R1*R2)":         "√(R1·R2) (media geom.)",
    "F1 (secondaria)":     "F1 (secondaria)",
    "Sensitivita' R1":     "Sensitività (R1)",
    "Evasion rate (1-R1)": "Evasion rate (1 − R1)",
}

# =============================================================================
# 4. STAMPA CONSOLE DEI VALORI (per copia nel testo, senza rileggere il JSON)
# =============================================================================
print(f"\n=== Outcome appaiato | n = {N_FRAME} | B = {N_ITER} ===")
print(f"{'metrica':<24}{'pre':>10}{'post':>10}{'Delta [IC 95%]':>26}  sig")
for n in names:
    pre, post, d = METRICS[n]["pre"], METRICS[n]["post"], METRICS[n]["paired_delta"]
    print(f"{LABEL.get(n, n):<24}{pre['point']:>10.3f}{post['point']:>10.3f}"
          f"  {d['delta']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
          f"  {'*' if d['significant'] else ''}")
print("(non tracciate, da citare nel testo: " + ", ".join(DEGENERI) + ")\n")

# =============================================================================
# 5. COSTRUZIONE DELLA FIGURA
# =============================================================================
n   = len(names)
y   = np.arange(n)[::-1]      # asse y categorico, prima metrica in alto
OFF = 0.20                    # sfalsamento verticale pre/post nel pannello (a)

# Palette: due tinte fisse per condizione nel (a); tinta neutra unica nel (b).
COL_PRE, COL_POST = "#1f77b4", "#ff7f0e"
COL_DELTA, COL_NULL = "#1a1a1a", "#888888"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 0.85 * n + 1.2), sharey=True)

# --- Pannello (a): livelli pre e post ---------------------------------------
for yi, nm in zip(y, names):
    pre, post = METRICS[nm]["pre"], METRICS[nm]["post"]
    axA.errorbar(pre["point"], yi + OFF,
                 xerr=[[pre["point"] - pre["low"]], [pre["high"] - pre["point"]]],
                 fmt="s", color=COL_PRE, ecolor=COL_PRE,
                 elinewidth=1.8, capsize=5, markersize=6, zorder=3,
                 label="pre" if yi == y[0] else "")
    axA.errorbar(post["point"], yi - OFF,
                 xerr=[[post["point"] - post["low"]], [post["high"] - post["point"]]],
                 fmt="o", color=COL_POST, ecolor=COL_POST,
                 elinewidth=1.8, capsize=5, markersize=6, zorder=3,
                 label="post" if yi == y[0] else "")

axA.set_xlim(-0.02, 1.02)
axA.set_xlabel("Stima [IC 95%]")
axA.set_title("(a)  Stime puntuali pre e post", loc="left")
axA.grid(axis="x", ls=":", alpha=0.5)
# Legenda in alto a destra: e' l'unica regione del pannello priva di marker,
# quindi non puo' sovrapporsi ai dati (in particolare all'evasion rate).
axA.legend(loc="upper right", frameon=False)

# --- Pannello (b): forest plot del Delta, senza annotazioni numeriche -------
for yi, nm in zip(y, names):
    d = METRICS[nm]["paired_delta"]
    point, lo, hi = d["delta"], d["low"], d["high"]
    axB.errorbar(point, yi,
                 xerr=[[point - lo], [hi - point]],
                 fmt="D", color=COL_DELTA, ecolor=COL_DELTA,
                 elinewidth=2.0, capsize=6, markersize=7, zorder=3)

# Asse simmetrico rispetto alla nullita': la linea a 0 cade al centro.
lows  = [METRICS[nm]["paired_delta"]["low"]  for nm in names]
highs = [METRICS[nm]["paired_delta"]["high"] for nm in names]
sym = float(np.ceil(max(abs(min(lows)), abs(max(highs))) / 0.05) * 0.05)
axB.set_xlim(-sym, sym)
axB.xaxis.set_major_locator(MultipleLocator(0.1))
axB.axvline(0, color=COL_NULL, lw=1.2, ls="--", zorder=1)
axB.set_xlabel("Δ (post − pre) [IC 95%]")
axB.set_title("(b)  Differenza appaiata post − pre", loc="left")
axB.grid(axis="x", ls=":", alpha=0.5)

# --- Asse y condiviso: etichette una sola volta, a sinistra -----------------
axA.set_yticks(y)
axA.set_yticklabels([LABEL.get(nm, nm) for nm in names])
axA.set_ylim(-0.6, n - 0.4)
for ax in (axA, axB):
    ax.tick_params(axis="y", length=0)     # nessuna tacca sull'asse categorico
axB.tick_params(labelleft=False)           # nessuna etichetta y duplicata sul (b)

fig.tight_layout()

# =============================================================================
# 6. SALVATAGGIO (raster ad alta risoluzione + vettoriale per la tesi)
# =============================================================================
_stem = Path(PATH_REPORT).stem                       # es. "full_report_okutama_960_stride27"
_suffix = _stem.replace("full_report", "").lstrip("_") or "output"
OUT_PNG = PROJECT_ROOT / "outputs" / "metrics" / f"fig_candele_{_suffix}.png"
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Salvato: {OUT_PNG}")
plt.show()

# =============================================================================
# 7. TESTI PER LA TESI (caption, metodi, nota VisDRONE)
# =============================================================================
# Coerenti con la figura rivista: il pannello (b) non contiene numeri ne'
# asterischi, pertanto il caption rimanda esplicitamente alla Tabella per i
# valori e dichiara la regola di lettura della nullita'.

CAPTION_IT = (
    "Figura X. Confronto pre/post delle metriche di outcome "
    f"(n = {N_FRAME} frame). I punti indicano le stime puntuali e le barre di "
    f"errore gli intervalli di confidenza al 95% ottenuti mediante bootstrap "
    f"non parametrico (B = {N_ITER} repliche). (a) Stime puntuali nelle due "
    "condizioni, pre (quadrati) e post (cerchi). (b) Forest plot della "
    "differenza appaiata Δ = post − pre; la linea verticale tratteggiata "
    "rappresenta l'ipotesi di nullità (Δ = 0). In ogni caso l'intervallo di "
    "confidenza non interseca tale linea, in accordo con i test appaiati i cui "
    "valori numerici e p-value sono riportati in Tabella X. Specificità R2 e "
    "danno collaterale medio per frame, risultati identici nelle due condizioni "
    "(Δ = 0, IC = [0, 0]), non sono rappresentati in quanto privi di variabilità "
    "bootstrap e sono discussi nel testo."
)

PARAGRAFO_METODO_IT = (
    "Per ciascuna metrica di outcome si riporta la stima puntuale osservata, "
    "accompagnata dall'intervallo di confidenza al 95% ottenuto con bootstrap "
    f"non parametrico ({N_ITER} repliche), mediante rappresentazione a punto e "
    "barre di errore. Essendo il confronto condotto con disegno appaiato sugli "
    "stessi frame, l'effetto del trattamento e' valutato sulla differenza "
    "appaiata Δ = post − pre: il relativo intervallo di confidenza, calcolato "
    "sulle differenze entro ciascun frame, e' piu' stretto di quello che si "
    "otterrebbe dal confronto di due intervalli indipendenti e costituisce il "
    "riferimento corretto per il giudizio di significativita'. Tale giudizio e' "
    "letto sul forest plot mediante la regola 'intervallo che non interseca lo "
    "zero ⇔ effetto significativo', in coerenza con il p-value del test "
    "appaiato riportato in tabella; il pannello dei livelli assoluti ha valore "
    "puramente descrittivo."
)

NOTA_VISDRONE_IT = (
    "Nota. Un'analoga stima bootstrap condotta sul sotto-campione VisDRONE "
    "(n = 89 frame) e' stata esclusa dalla figura principale per l'insufficiente "
    "numerosita' campionaria, che renderebbe gli intervalli poco informativi; "
    "tale analisi e' conservata, ove richiesta, come verifica di robustezza in "
    "appendice."
)

# print("\n--- CAPTION ---\n", CAPTION_IT)
# print("\n--- METODI ---\n", PARAGRAFO_METODO_IT)
# print("\n--- NOTA VisDRONE ---\n", NOTA_VISDRONE_IT)