#!/usr/bin/env python3
"""
Command-line interface for LAWS-SIM.
"""

import argparse
import json
import sys

from rich.table import Table
from config import PATCH_STEPS, EOT_N_TRANSFORMS
from metrics import AttackScenario, SimMetrics, clae_costs, compute_clae
from simulator import LAWSSim
from patch_optimizer import PatchOptimizer
from utils import console, HAS_RICH, HAS_MPL, save_patch_plots


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_demo_patch(image_path: str, model_path: str = "yolov8n.pt", steps: int = PATCH_STEPS):
    """Run adversarial patch demo (standalone)."""
    if not HAS_TORCH:
        console.print("[red]Demo requires: pip install torch ultralytics[/red]")
        return

    console.print(f"[bold cyan]Demo Adversarial Patch — EoT[/bold cyan]")
    console.print(f"Image: [yellow]{image_path}[/yellow]")
    console.print(f"Model: [yellow]{model_path}[/yellow]")
    console.print(f"Optimization steps: [yellow]{steps}[/yellow]")
    console.print(f"EoT transforms/step: [yellow]{EOT_N_TRANSFORMS}[/yellow]")

    optimizer = PatchOptimizer(model_path=model_path)
    result = optimizer.optimize(image_path, n_steps=steps, verbose=True)

    cov = result["patch_coverage"]
    delta_conf = result["conf_drop"]
    clae_real = delta_conf / cov if cov > 0 else 0.0

    console.print(f"\n[bold green]Summary[/bold green]")
    console.print(f"Confidence before: [green]{result['conf_before']:.4f}[/green]")
    console.print(f"Confidence after:  [red]{result['conf_after']:.4f}[/red]")
    console.print(f"Drop: [bold yellow]{delta_conf:+.4f} ({delta_conf/max(result['conf_before'],1e-6)*100:.1f}%)[/bold yellow]")
    console.print(f"C_vision (pixel ratio) = {cov:.4f} ({cov*100:.1f}% of bbox)")
    console.print(f"CLAE = Δconf / C_vision = [bold cyan]{clae_real:.3f}[/bold cyan]")

    if HAS_MPL:
        patch_arr = (result["patch"].permute(1, 2, 0).numpy() * 255).astype("uint8")
        result["patch_arr"] = patch_arr
        save_patch_plots(result, output_dir=".")

    torch.save(result["patch"], "care_kit_patch.pt")
    console.print("[green]✓ Patch saved → care_kit_patch.pt[/green]")


def print_results(results: dict, baseline: SimMetrics):
    """Print simulation results table."""
    costs = clae_costs()
    if not HAS_RICH:
        for sc, m in results.items():
            clae = compute_clae(sc, m, baseline)
            print(f"  {sc.value:<42} F1={m.f1:.3f}  CLAE={'—' if clae is None else f'{clae:.3f}'}")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title="[bold cyan]LAWS-SIM v3 — Results[/bold cyan]")
    for col in ["Scenario", "Precision", "Recall", "F1", "FPR", "CLAE v2"]:
        table.add_column(col, justify="right" if col != "Scenario" else "left")

    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        cs = "—" if clae is None else f"{clae:.3f}"
        color = "green" if m.f1 > 0.60 else ("yellow" if m.f1 > 0.35 else "red")
        table.add_row(sc.value, f"{m.precision:.3f}", f"{m.recall:.3f}",
                      f"[{color}]{m.f1:.3f}[/{color}]", f"{m.fpr:.3f}", cs)
    console.print(table)

    c_v, c_o, c_c = costs["PATCH_ONLY"], costs["OSINT_POISON"], costs["CASCADING"]
    console.print(f"[bold]CLAE v2 — measurable costs[/bold]")
    console.print(f"C_vision = {c_v:.2f} (pixel ratio)")
    console.print(f"C_osint = {c_o:.2f} (poisoned/total fields)")
    console.print(f"C_cascade = {c_c:.4f} (union probability)")


def main():
    parser = argparse.ArgumentParser(description="LAWS-SIM v3.0")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--real-yolo", action="store_true")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--demo-patch", type=str, default=None, metavar="IMAGE")
    parser.add_argument("--patch", type=str, default=None)
    parser.add_argument("--patch-steps", type=int, default=PATCH_STEPS)
    args = parser.parse_args()

    if args.demo_patch:
        run_demo_patch(args.demo_patch, steps=args.patch_steps)
        return

    patch_tensor = None
    if args.patch and HAS_TORCH:
        patch_tensor = torch.load(args.patch, weights_only=False)
        console.print(f"[green]✓ Patch loaded: {args.patch} ({patch_tensor.shape})[/green]")

    console.print(f"[bold cyan]LAWS-SIM v3.0[/bold cyan]")
    console.print(f"Steps={args.steps} Seed={args.seed} RealYOLO={'✓' if args.real_yolo else '✗'} Patch={'✓' if patch_tensor is not None else '✗'}")

    results = {}
    for sc in AttackScenario:
        console.print(f"Simulating: {sc.value}…")
        sim = LAWSSim(sc, args.steps, args.seed,
                      real_mode=args.real_yolo,
                      image_dir=args.image_dir,
                      patch_tensor=patch_tensor)
        results[sc] = sim.run(verbose=args.verbose)

    baseline = results[AttackScenario.NONE]
    print_results(results, baseline)

    export = {}
    costs = clae_costs()
    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        export[sc.value] = {
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "fpr": round(m.fpr, 4),
            "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
            "clae_v2": round(clae, 4) if clae else None,
            "cost_measurable": round(costs.get(sc.name) or 0, 4)
        }
    with open("laws_sim_v3_results.json", "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    console.print("[green]✓ Export → laws_sim_v3_results.json[/green]")


if __name__ == "__main__":
    main()