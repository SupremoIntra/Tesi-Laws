"""
Grafico dell'andamento di Loss principale e TV Loss durante il training,
letto da outputs/metrics/training_metrics.json.

Uso:
    python tools/plot_training_curves.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
from config import METRICS_JSON_FILE, METRICS_DIR


def main():
    with open(METRICS_JSON_FILE, "r") as f:
        data = json.load(f)

    steps = data["step"]
    loss = data["loss"]
    tv_loss = data["tv_loss"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, loss, color="#c0392b", linewidth=1)
    ax1.axhline(y=0.6931, color="gray", linestyle="--", linewidth=1, label="ln(2) — punto neutro (mean_conf=0.5)")
    ax1.set_ylabel("Loss principale (asintotica)")
    ax1.set_title("Andamento Loss principale vs TV Loss durante il training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, tv_loss, color="#2980b9", linewidth=1)
    ax2.set_ylabel("TV Loss")
    ax2.set_xlabel("Step raw")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(METRICS_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Grafico salvato in: {out_path}")


if __name__ == "__main__":
    main()
