"""
Confronto F1/Evasion Rate PUNTUALE (senza CI) tra le 6 configurazioni
storiche di training — documenta la RICERCA DEGLI IPERPARAMETRI (perché
K=20, perché accum=4), non la significatività dell'attacco finale.
Per quella, vedi cli.py --eval-report (bootstrap + delta appaiato + p-value).


Uso:
    python tools/plot_runs_comparison.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Aggiornare ad ogni nuovo run completato (stessa fonte della tabella in thesis_notes.md)
RUNS = [
    {"name": "Baseline", "f1": 0.760, "evasion": 38.7, "upd": 3000},
    {"name": "Run 1", "f1": 0.740, "evasion": 41.25, "upd": 375},
    {"name": "Run 2", "f1": 0.750, "evasion": 40.0, "upd": 750},
    {"name": "Fase 1", "f1": 0.720, "evasion": 43.75, "upd": 2500},
    {"name": "Fase 2", "f1": 0.740, "evasion": 41.25, "upd": 625},
    {"name": "Fase 3", "f1": 0.720, "evasion": 43.75, "upd": 1250},
]


def main():
    names = [r["name"] for r in RUNS]
    f1_vals = [r["f1"] for r in RUNS]
    evasion_vals = [r["evasion"] for r in RUNS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bars1 = ax1.bar(names, f1_vals, color="#2980b9")
    ax1.set_ylabel("F1-Score")
    ax1.set_title("F1-Score sotto attacco")
    ax1.set_ylim(0, 1.0)
    for b, v, r in zip(bars1, f1_vals, RUNS):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        ax1.text(b.get_x() + b.get_width() / 2, -0.06, f"{r['upd']} upd", ha="center", fontsize=8, color="gray")

    bars2 = ax2.bar(names, evasion_vals, color="#c0392b")
    ax2.set_ylabel("Evasion Rate (%)")
    ax2.set_title("Evasion Rate — soffitto strutturale ~44%")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=max(evasion_vals), color="gray", linestyle="--", linewidth=1)
    for b, v, r in zip(bars2, evasion_vals, RUNS):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center")
        ax2.text(b.get_x() + b.get_width() / 2, -6, f"{r['upd']} upd", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    out_dir = "outputs/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "runs_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Grafico salvato in: {out_path}")


if __name__ == "__main__":
    main()
