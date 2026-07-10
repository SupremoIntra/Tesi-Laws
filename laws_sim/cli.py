"""
CLI per LAWS-SIM
Gestisce la valutazione empirica su VisDrone e la successiva simulazione multi-agente.
"""
import argparse
import os
import sys
import json

# Deve stare prima di "import torch": grid_sampler_2d_backward non è
# implementato su MPS, questo fa girare solo quell'operatore su CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# Percorso di ricerca moduli: src/ contiene i moduli applicativi, aggiunto
# al path invece di riscrivere tutti gli import interni (es. "from config
# import ..." in patch_optimizer.py) — minimizza le modifiche al codice
# già validato.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import (
    PATCH_BBOX_COVERAGE, OSINT_FIELDS_POISONED, OSINT_FIELDS_TOTAL,
    VISION_METRICS_JSON, BEST_PATCH_FILE
)
from metrics import AttackScenario, compute_clae
from utils import console

def main():
    parser = argparse.ArgumentParser(description="LAWS-SIM: Autonomous Weapons Security Framework")
    parser.add_argument("--train-patch", action="store_true", help="Addestra la Universal Patch da zero")
    parser.add_argument("--train-dir", type=str, default="data/visdrone_train", metavar="DIR",
                         help="Cartella VisDrone (trainset) per l'ottimizzazione della patch (default: data/visdrone_train)")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignora/rimuove il checkpoint esistente e riparte da zero (usare quando si cambiano iperparametri di training)")
    parser.add_argument("--eval-vision", type=str, metavar="DIR", help="Cartella dataset VisDrone (valset) per validare YOLO")
    parser.add_argument("--patch", type=str, metavar="FILE", help="Percorso del tensore patch (.pt)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limita i frame di test (VisDrone)")
    parser.add_argument("--run-sim", action="store_true", help="Esegue il simulatore multi-agente disaccoppiato")
    args = parser.parse_args()

    # FASE 0: Addestramento Universale
    if args.train_patch:
        console.print("\n[bold green]Avvio Addestramento Universal Patch (SOTA Architecture)[/bold green]")
        try:
            from visdrone_loader import VisDroneLoader
            from patch_optimizer import PatchOptimizer
            from config import CHECKPOINT_FILE
        except ImportError as e:
            console.print(f"[red]Errore dipendenze: {e}[/red]")
            return

        if args.fresh and os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            console.print(f"[yellow]--fresh: rimosso checkpoint precedente ({CHECKPOINT_FILE})[/yellow]")

        loader = VisDroneLoader(args.train_dir)

        # TACTICAL FILTER 2026: pre-flight check obbligatorio prima del training
        from patch_optimizer import tactical_preflight_check
        tactical_preflight_check(loader, low=60.0, high=80.0)

        optimizer = PatchOptimizer(model_path="yolov8n.pt")

        try:
            results = optimizer.optimize_universal(loader=loader, verbose=True)
            torch.save(results["patch"], BEST_PATCH_FILE)
            console.print(f"\n[bold green]✓ Universal Patch salvata in: {BEST_PATCH_FILE}[/bold green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Addestramento interrotto manualmente.[/yellow]")
        return

    # FASE 1: Validazione Empirica del Sensore (VisDrone + YOLO)
    if args.eval_vision:
        if not args.patch:
            console.print("[red]Errore: Devi fornire la patch con --patch per valutare sotto attacco.[/red]")
            return

        console.print(f"\n[bold cyan]Avvio Validazione Visiva su: {args.eval_vision}[/bold cyan]")
        try:
            from visdrone_loader import VisDroneLoader
            from simulator import evaluate_on_dataset
        except ImportError as e:
            console.print(f"[red]Errore importazione PyTorch/YOLO: {e}[/red]")
            return

        loader = VisDroneLoader(args.eval_vision)
        console.print(f"VisDroneLoader: {len(loader)} frame validi.")

        pt_path = args.patch
        try:
            patch_tensor = torch.load(pt_path, map_location="cpu")
            console.print(f"Test in corso CON patch applicata: {pt_path}")
        except Exception as e:
            console.print(f"[red]Errore caricamento patch: {e}[/red]")
            return

        # Eseguo il test su YOLO e ottengo i risultati
        # TACTICAL FILTER 2026: ora restituisce anche le metriche filtrate
        # (>=80px) e la copertura tattica del valset, calcolate a costo zero
        # nello stesso loop di inferenza.
        metrics, metrics_tactical, tactical_coverage = evaluate_on_dataset(
            loader=loader,
            patch_tensor=patch_tensor,
            max_samples=args.max_samples,
            verbose=False
        )
        filtered_evasion_rate = metrics_tactical.fn / max(metrics_tactical.tp + metrics_tactical.fn, 1)

        # Salvo il risultato empirico (F1-Score) per il simulatore
        results = {
            "f1": metrics.f1, "precision": metrics.precision, "recall": metrics.recall,
            "filtered_evasion_rate": filtered_evasion_rate,  # TACTICAL FILTER 2026
            "tactical_coverage": tactical_coverage,          # TACTICAL FILTER 2026
        }
        os.makedirs(os.path.dirname(VISION_METRICS_JSON), exist_ok=True)
        with open(VISION_METRICS_JSON, "w") as f:
            json.dump(results, f)

        # Stampa i risultati
        from rich.table import Table
        t = Table(title="Risultati Empirici Visione (VisDrone) - SOTTO ATTACCO", style="cyan")
        t.add_column("Metrica", style="bold")
        t.add_column("Valore", justify="right")
        t.add_row("F1-Score (completo)", f"{metrics.f1:.3f}")
        t.add_row("Precision", f"{metrics.precision:.3f}")
        t.add_row("Recall", f"{metrics.recall:.3f}")
        t.add_row("True Positives (TP)", str(metrics.tp))
        t.add_row("False Negatives (FN)", str(metrics.fn))
        t.add_row("Evasion Rate (completo)", f"{(metrics.fn / max(metrics.tp + metrics.fn, 1)) * 100:.1f}%")
        t.add_row("Evasion Rate (target >=80px)", f"{filtered_evasion_rate * 100:.1f}%")
        t.add_row("Copertura tattica valset (>=80px)", f"{tactical_coverage * 100:.1f}%")
        console.print(t)
        console.print(f"[green]✓ Metriche visive salvate per il simulatore ({VISION_METRICS_JSON}).[/green]\n")

    # FASE 2: Simulazione Tattica Multi-Agente
    if args.run_sim:
        from simulator import LAWSSim
        console.print("\n[bold magenta]Avvio Simulatore Multi-Agente (LAWS-SIM)[/bold magenta]")

        if os.path.exists(VISION_METRICS_JSON):
            with open(VISION_METRICS_JSON, "r") as f:
                d = json.load(f)
                console.print(f"[green]✓ Dati visivi empirici caricati (F1={d.get('f1', 0):.3f})[/green]")
        else:
            console.print("[yellow]! Nessun dato empirico trovato. Uso F1 Baseline (0.710).[/yellow]")

        from rich.table import Table
        results = {}

       # ==========================================
        # ESPERIMENTO 1: ISOLAMENTO DOMINIO VISIVO
        # ==========================================
        console.print("\n[bold cyan]ESPERIMENTO 1: Impatto Diretto sul Sensore Visivo (Vision-Only)[/bold cyan]")
        t1 = Table(title="Dominio Visivo Puro (Pesi: Vision 100%, OSINT 0%, Behavior 0%)", style="cyan")
        t1.add_column("Scenario", style="bold")
        t1.add_column("Target (TP)", justify="right", style="green")
        t1.add_column("Civili (FP)", justify="right", style="red")
        t1.add_column("Precision")
        t1.add_column("Recall")
        t1.add_column("F1-Score", style="bold")
        t1.add_column("CEAE")

        for s in [AttackScenario.BASELINE, AttackScenario.PATCH_ONLY]:
            sim = LAWSSim(scenario=s, steps=150)

            # HACK NARRATIVO: Spegniamo l'OSINT per isolare la vera potenza della patch
            sim.fusion.w = {"vision": 1.0, "osint": 0.0, "behavioral": 0.0}

            m = sim.run(verbose=False)
            c_vision = PATCH_BBOX_COVERAGE if s == AttackScenario.PATCH_ONLY else 0.0
            ceae_val = compute_clae(m, c_vision, 0.0)

            t1.add_row(s.value, str(m.tp), str(m.fp), f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}", f"{ceae_val:.3f}")

        console.print(t1)

        # ==========================================
        # ESPERIMENTO 2: VULNERABILITA' MULTI-DOMINIO
        # ==========================================
        console.print("\n[bold magenta]ESPERIMENTO 2: Sensore Fusion e Guerra Cibernetica[/bold magenta]")
        t2 = Table(title="Sistema Completo (Pesi standard: Vision 45%, OSINT 35%, Behavior 20%)", style="magenta")
        t2.add_column("Scenario", style="bold")
        t2.add_column("Target (TP)", justify="right", style="green")
        t2.add_column("Civili (FP)", justify="right", style="red")
        t2.add_column("Precision")
        t2.add_column("Recall")
        t2.add_column("F1-Score", style="bold")
        t2.add_column("FPR")
        t2.add_column("CEAE")

        for s in [AttackScenario.BASELINE, AttackScenario.PATCH_ONLY, AttackScenario.OSINT_POISONING, AttackScenario.CASCADING]:
            sim = LAWSSim(scenario=s, steps=150)

            # Scenario standard: usa i pesi di default definiti in FUSION_WEIGHTS
            m = sim.run(verbose=False)

            c_vision = PATCH_BBOX_COVERAGE if s in (AttackScenario.PATCH_ONLY, AttackScenario.CASCADING) else 0.0
            c_osint = (OSINT_FIELDS_POISONED / OSINT_FIELDS_TOTAL) if s in (AttackScenario.OSINT_POISONING, AttackScenario.CASCADING) else 0.0
            ceae_val = compute_clae(m, c_vision, c_osint)

            t2.add_row(s.value, str(m.tp), str(m.fp), f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}", f"{m.fpr:.3f}", f"{ceae_val:.3f}")

        console.print(t2)
        console.print("[green]✓ Test completati con successo.[/green]\n")

if __name__ == "__main__":
    main()
