"""
Utility functions (console output, plotting).
"""
import numpy as np
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    class _FC:
        def print(self, *a, **kw): print(*[str(x) for x in a])
    console = _FC()
    HAS_RICH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def save_patch_plots(result_dict: dict, output_dir: str = "."):
    """Save visual results of adversarial patch optimization."""
    if not HAS_MPL:
        return
    out = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Adversarial Patch — EoT Results", fontsize=13, fontweight="bold")

    orig_np = (result_dict["img_original"].permute(1, 2, 0).numpy() * 255).astype("uint8")
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original\nconf person = {result_dict['conf_before']:.4f}", color="green")
    x1, y1, x2, y2 = result_dict["bbox"]
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].axis("off")

    axes[1].imshow(result_dict.get("patch_arr", np.zeros((100, 80, 3))))
    cov = result_dict["patch_coverage"]
    axes[1].set_title(f"Optimized Patch (EoT)\ncoverage = {cov:.3f} → C_vision", color="orange")
    axes[1].axis("off")

    patched_np = (result_dict["img_patched"].permute(1, 2, 0).numpy() * 255).astype("uint8")
    axes[2].imshow(patched_np)
    axes[2].set_title(f"With Patch Applied\nconf person = {result_dict['conf_after']:.4f}", color="red")
    rect2 = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
    axes[2].add_patch(rect2)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out / "patch_result.png", dpi=150, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result_dict["loss_history"], color="#e74c3c", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss (mean person confidence)")
    ax.set_title("EoT Convergence — Adversarial Patch Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "patch_loss.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    console.print(f"[green]Plots saved → {out}/patch_result.png, patch_loss.png[/green]")