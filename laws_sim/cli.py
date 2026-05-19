#!/usr/bin/env python3
"""
CLI - LAWS-SIM Framework
Gestione centralizzata per Addestramento Patch, Valutazione Visiva (VisDrone) e Simulazione Multi-Agente.
"""

import os
# Il fallback per Mac M4 deve essere caricato PRIMA di importare torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import sys

from rich.table import Table
from config import PATCH_STEPS
from metrics import AttackScenario, SimMetrics, clae_costs, compute_clae
from simulator import LAWSSim, evaluate_on_dataset
from patch_optimizer import PatchOptimizer
from visdrone_loader import VisDroneLoader
from utils import console, HAS_RICH

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def print_vision_results(metrics, is_baseline=True):
    """Stampa la tabella accademica dei risultati puramente visivi (VisDrone)."""
    title = "[bold cyan]Risultati Empirici Visione (VisDrone) - BASELINE[/bold cyan]" if is_baseline else "[bold red]Risultati Empirici Visione (VisDrone) - SOTTO ATTACCO[/bold red]"
    
    table = Table(show_header=True, header_style="bold magenta", title=title)
    table.add_column("Metrica", justify="left")
    table.add_column("Valore", justify="right")

    color = "green" if metrics.f1 > 0.50 else "red"
    
    table.add_row("F1-Score", f"[{color}]{metrics.f1:.3f}[/{color}]")
    table.add_row("Precision", f"{metrics.precision:.3f}")
    table.add_row("Recall", f"{metrics.recall:.3f}")
    table.add_row("True Positives (TP)", str(metrics.tp))
    table.add_row("False Positives (FP)", str(metrics.fp))
    table.add_row("False Negatives (FN)", str(metrics.fn))
    
    console.print(table)


def print_results(results: dict, baseline: SimMetrics):
    """Stampa la tabella dei risultati e metriche dell'ambiente simulato (CEAE)."""
    costs = clae_costs()
    
    table = Table(show_header=True, header_style="bold magenta",
                  title="[bold cyan]LAWS-SIM Framework — RISULTATI DI SISTEMA (Multi-Agente)[/bold cyan]")
    for col in ["Scenario", "Precision", "Recall", "F1 (P+R)", "FPR", "CEAE"]:
        table.add_column(col, justify="right" if col != "Scenario" else "left")

    for sc, m in results.items():
        ceae = compute_clae(sc, m, baseline)
        cs = "—" if ceae is None else f"{ceae:.3f}"
        color = "green" if m.f1 > 0.60 else ("yellow" if m.f1 > 0.35 else "red")
        table.add_row(sc.value, f"{m.precision:.3f}", f"{m.recall:.3f}",
                      f"[{color}]{m.f1:.3f}[/{color}]", f"{m.fpr:.3f}", cs)
    console.print(table)

    c_v, c_o, c_c = costs["PATCH_ONLY"], costs["OSINT_POISON"], costs["CASCADING"]
    console.print("\n[bold]CEAE — Costi Misurabili[/bold]")
    console.print(f"C_vision  = {c_v:.2f} (percentuale pixel patch su bbox)")
    console.print(f"C_osint   = {c_o:.2f} (campi avvelenati / campi totali)")
    console.print(f"C_cascade = {c_c:.4f} (union probability)")


def main():
    parser = argparse.ArgumentParser(description="LAWS-SIM - Framework Tesi")
    
    # 1. Modulo Addestramento Patch
    parser.add_argument("--train-universal", type=str, default=None, metavar="DIR", help="Addestra Universal Patch su VisDrone")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per Universal Patch")
    
    # 2. Modulo Valutazione Visiva
    parser.add_argument("--eval-vision", type=str, default=None, metavar="DIR", help="Valuta YOLO su VisDrone")
    parser.add_argument("--patch", type=str, default=None, metavar="FILE", help="Percorso file .pt (es. care_kit_patch_universal.pt)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limite frame (test rapidi)")
    
    # 3. Modulo Simulatore Multi-Agente
    parser.add_argument("--run-sim", action="store_true", help="Avvia simulatore multi-agente per calcolo CEAE")
    parser.add_argument("--steps", type=int, default=150, help="Step temporali del simulatore")
    
    # Utilità
    parser.add_argument("--verbose", action="store_true", help="Stampa log dettagliati")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    if not (args.train_universal or args.eval_vision or args.run_sim):
        parser.print_help()
        return

    # --- FASE 1: ADDESTRAMENTO ---
    if args.train_universal:
        console.print(f"[bold cyan]Avvio Training Universal Patch su: {args.train_universal}[/bold cyan]")
        loader = VisDroneLoader(args.train_universal)
        opt = PatchOptimizer()
        res = opt.optimize_universal(loader, n_steps=PATCH_STEPS, batch_size=args.batch_size, verbose=args.verbose)
        
        torch.save(res["patch"], "care_kit_patch_universal.pt")
        console.print("[green]✓ Patch Universale salvata in care_kit_patch_universal.pt[/green]")
        return  # Si ferma qui, il training impiega ore.

    # --- FASE 2: VALUTAZIONE VISIONE ---
    if args.eval_vision:
        console.print(f"[bold cyan]Avvio Valutazione Visiva su: {args.eval_vision}[/bold cyan]")
        loader = VisDroneLoader(args.eval_vision)
        
        pt_tensor = None
        if args.patch and HAS_TORCH:
            pt_tensor = torch.load(args.patch, weights_only=False)
            console.print(f"Test in corso CON patch applicata: {args.patch}")
        else:
            console.print("Test in corso SENZA patch (Baseline Pura).")
            
        metrics = evaluate_on_dataset(loader, patch_tensor=pt_tensor, max_samples=args.max_samples, verbose=args.verbose)
        
        print_vision_results(metrics, is_baseline=(pt_tensor is None))
        
        # Salvataggio silente (IL PONTE VERSO IL SIMULATORE)
        vision_data = {
            "f1": metrics.f1, "precision": metrics.precision,
            "recall": metrics.recall, "patch_applied": pt_tensor is not None
        }
        with open("vision_metrics.json", "w") as f:
            json.dump(vision_data, f, indent=2)
        console.print("[dim]✓ Metriche visive salvate per il simulatore.[/dim]\n")

    # --- FASE 3: SIMULATORE MULTI-AGENTE ---
    if args.run_sim:
        console.print(f"[bold cyan]Avvio Simulatore Multi-Agente (LAWS-SIM)[/bold cyan]")
        
        # Carica automaticamente l'F1 empirico
        if os.path.exists("vision_metrics.json"):
            with open("vision_metrics.json", "r") as f:
                vision_data = json.load(f)
            console.print(f"[green]✓ Dati visivi empirici caricati (F1={vision_data.get('f1'):.3f})[/green]")
        else:
            console.print("[yellow]⚠ Nessun file vision_metrics.json trovato. Avvio con logiche interne default.[/yellow]")

        patch_tensor = None
        if args.patch and HAS_TORCH:
            patch_tensor = torch.load(args.patch, weights_only=False)

        results = {}
        for sc in AttackScenario:
            console.print(f"Simulazione in corso: {sc.value}…")
            # NOTA: Nel prossimo giro aggiorneremo LAWSSim per leggere effettivamente il JSON
            sim = LAWSSim(sc, args.steps, args.seed, patch_tensor=patch_tensor)
            results[sc] = sim.run(verbose=args.verbose)

        baseline = results[AttackScenario.NONE]
        print_results(results, baseline)

        # Export
        export = {}
        costs = clae_costs()
        for sc, m in results.items():
            ceae = compute_clae(sc, m, baseline)
            export[sc.value] = {
                "precision": round(m.precision, 4), "recall": round(m.recall, 4),
                "f1": round(m.f1, 4), "fpr": round(m.fpr, 4),
                "CEAE": round(ceae, 4) if ceae else None
            }
        with open("laws_sim_results.json", "w") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        console.print("[green]✓ Export completato → laws_sim_results.json[/green]")


if __name__ == "__main__":
    main()