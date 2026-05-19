#!/usr/bin/env python3
"""
CLI - traduzione parametri da terminale
"""

import argparse
import json
import sys

#import per dataset visdrone
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Deve essere caricato PRIMA di torch
from visdrone_loader import VisDroneLoader

from rich.table import Table
from config import PATCH_STEPS, EOT_N_TRANSFORMS
from metrics import AttackScenario, SimMetrics, clae_costs, compute_clae
from simulator import LAWSSim, evaluate_on_dataset
from patch_optimizer import PatchOptimizer
from utils import console, HAS_RICH, HAS_MPL, save_patch_plots


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def print_results(results: dict, baseline: SimMetrics):
    """Stampo la tabella dei risultati e metriche dell'ambiente simulato."""
    costs = clae_costs()
    if not HAS_RICH:
        for sc, m in results.items():
            clae = compute_clae(sc, m, baseline)
            print(f"  {sc.value:<42} F1={m.f1:.3f}  CLAE={'—' if clae is None else f'{clae:.3f}'}")
        return

    table = Table(show_header=True, header_style="bold magenta",
                  title="[bold cyan]LAWS-SIM Framework — RISULTATI[/bold cyan]")
    for col in ["Scenario", "Precision", "Recall", "F1 (P+R)", "FPR (False Positive Rate)", "CEAE"]:
        table.add_column(col, justify="right" if col != "Scenario" else "left")

    for sc, m in results.items():
        clae = compute_clae(sc, m, baseline)
        cs = "—" if clae is None else f"{clae:.3f}"
        color = "green" if m.f1 > 0.60 else ("yellow" if m.f1 > 0.35 else "red")
        table.add_row(sc.value, f"{m.precision:.3f}", f"{m.recall:.3f}",
                      f"[{color}]{m.f1:.3f}[/{color}]", f"{m.fpr:.3f}", cs)
    console.print(table)

    c_v, c_o, c_c = costs["PATCH_ONLY"], costs["OSINT_POISON"], costs["CASCADING"]
    console.print(f"[bold]CEAE — costi misurabili[/bold]")
    console.print(f"C_vision = {c_v:.2f} (percentuale pixel patch su bbox)")
    console.print(f"C_osint = {c_o:.2f} (campi avvelenati / campi totali)")
    console.print(f"C_cascade = {c_c:.4f} (union probability)")


def main():
    parser = argparse.ArgumentParser(description="LAWS-SIM")
    #  NUOVI COMANDI VISDRONE E UNIVERSAL PATCH 
    parser.add_argument("--eval-dataset", type=str, default=None, metavar="DIR", help="Calcola baseline o eval con patch su VisDrone")
    parser.add_argument("--train-universal", type=str, default=None, metavar="DIR", help="Addestra Universal Patch su VisDrone")
    parser.add_argument("--max-samples", type=int, default=None, help="Limite frame per eval-dataset (test rapidi)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per Universal Patch (salvaguardia VRAM M4)")
    #####
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

        # ADDESTRAMENTO UNIVERSAL PATCH 
    if args.train_universal:
        console.print(f"[bold cyan]Avvio Training Universal Patch su: {args.train_universal}[/bold cyan]")
        loader = VisDroneLoader(args.train_universal)
        opt = PatchOptimizer()
        res = opt.optimize_universal(loader, n_steps=args.patch_steps, batch_size=args.batch_size, verbose=args.verbose)
        
        # Salva fisicamente il file della patch universale
        torch.save(res["patch"], "care_kit_patch_universal.pt")
        console.print("[green]✓ Patch Universale salvata in care_kit_patch_universal.pt[/green]")
        return

    # VALUTAZIONE EMPIRICA (VISDRONE)
    if args.eval_dataset:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        console.print(f"[bold cyan]Avvio Valutazione Empirica su: {args.eval_dataset}[/bold cyan]")
        loader = VisDroneLoader(args.eval_dataset)
        
        # Se l'utente ha passato anche --patch, carichiamo il tensore per testare l'attacco
        pt_tensor = None
        if args.patch and HAS_TORCH:
            pt_tensor = torch.load(args.patch, weights_only=False)
            console.print(f"Test in corso CON patch applicata: {args.patch}")
        else:
            console.print("Test in corso SENZA patch (Baseline Pura).")
            
        metrics = evaluate_on_dataset(loader, patch_tensor=pt_tensor, max_samples=args.max_samples, verbose=True)
        
        console.print(f"\n[bold green]F1-Score Reale: {metrics.f1:.3f}[/bold green] (Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f})")
        return

    patch_tensor = None
    if args.patch and HAS_TORCH:
        patch_tensor = torch.load(args.patch, weights_only=False)
        console.print(f"[green]OK Patch caricata: {args.patch} ({patch_tensor.shape})[/green]")

    console.print(f"[bold cyan]LAWS-SIM[/bold cyan]")
    console.print(f"Steps={args.steps} Seed={args.seed} RealYOLO={'✓' if args.real_yolo else '✗'} Patch={'✓' if patch_tensor is not None else '✗'}")

    results = {}
    for sc in AttackScenario:
        console.print(f"Simulazione in corso: {sc.value}…")
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
            "CEAE": round(clae, 4) if clae else None,
            "cost_measurable": round(costs.get(sc.name) or 0, 4)
        }
    with open("laws_sim_results.json", "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    console.print("[green]✓ Export → laws_sim_results.json[/green]")


if __name__ == "__main__":
    main()