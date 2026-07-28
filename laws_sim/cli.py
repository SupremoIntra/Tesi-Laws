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
    parser.add_argument("--eval-report", action="store_true",
                         help="Report Vision completo pre/post in un comando: CI + delta appaiato + p-value (+ extra con --full-report)")
    parser.add_argument("--data", type=str, default="data/visdrone_val", metavar="DIR",
                         help="Valset per --eval-report (default: data/visdrone_val)")
    parser.add_argument("--n-iter", type=int, default=10000, help="Iterazioni bootstrap per --eval-report")
    parser.add_argument("--full-report", action="store_true",
                         help="Con --eval-report: aggiunge stratificazione per taglia, confidence drop e grafici K (piu' lento, rilancia i tool)")
    parser.add_argument("--conf-threshold", type=float, default=0.50,
                         help="Soglia di confidenza YOLO per --eval-report (default 0.50, per verifica di robustezza)")
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone",
                     help="Dataset loader per --train-patch (default: visdrone)")
    parser.add_argument("--img-size", type=int, default=960, help="Canvas per loader okutama")
    parser.add_argument("--patch-out", type=str, default=None, metavar="FILE",
                        help="Percorso di salvataggio patch per --train-patch (default: BEST_PATCH_FILE, sovrascrive care_kit_patch_universal.pt)")
    parser.add_argument("--stride", type=int, default=1,
                     help="Usa 1 frame ogni N (decorrelazione per dataset video, es. Okutama)")
    args = parser.parse_args()

    # ===== Report Vision consolidato (un comando, metriche richieste dal relatore) =====
    if args.eval_report:
        if not args.patch:
            console.print("[red]Errore: --eval-report richiede --patch.[/red]")
            return

        from visdrone_loader import VisDroneLoader
        from simulator import evaluate_on_dataset
        from metrics import (
            bootstrap_ci, paired_bootstrap_diff,
            evasion_rate_from_counts, sensitivity_from_counts,
            specificity_from_counts, gmean_from_counts, f1_from_counts,
        )
        from rich.table import Table

        from okutama_loader import OkutamaLoader
        LOADERS = {"visdrone": VisDroneLoader, "okutama": OkutamaLoader}
        loader = (LOADERS[args.loader](args.data, img_size=args.img_size)
                   if args.loader == "okutama" else LOADERS[args.loader](args.data))
        if args.stride > 1:
            loader.samples = loader.samples[::args.stride]
            console.print(f"[dim]Stride={args.stride}: {len(loader)} frame dopo decorrelazione[/dim]")
        patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)

        # Due sole passate di inferenza: PRE (senza patch) e POST (con patch),
        # stesso valset e stesso ordine -> per_frame_outcomes appaiabili.
        console.print("\n[bold cyan]PRE-attacco (YOLO naturale, nessuna patch)[/bold cyan]")
        _, _, _, outcomes_pre = evaluate_on_dataset(
            loader=loader, patch_tensor=None, max_samples=args.max_samples, verbose=False, conf_threshold=args.conf_threshold)
        console.print("[bold cyan]POST-attacco (con patch)[/bold cyan]")
        _, _, _, outcomes_post = evaluate_on_dataset(
            loader=loader, patch_tensor=patch_tensor, max_samples=args.max_samples, verbose=False, conf_threshold=args.conf_threshold)

        # Ordine di priorita' del relatore: evasion rate (efficacia patch) in testa,
        # poi R1/R2 e la loro media geometrica (metrica primaria), F1 in coda come
        # secondaria ("la usano tutti ma fa caos").
        metric_specs = [
            ("Evasion rate (1-R1)", evasion_rate_from_counts),
            ("Sensitivita' R1",     sensitivity_from_counts),
            ("Specificita' R2",     specificity_from_counts),
            ("sqrt(R1*R2)",         gmean_from_counts),
            ("F1 (secondaria)",     f1_from_counts),
        ]

        t = Table(title=f"Report Vision {args.loader.capitalize()} (n={len(outcomes_pre)} frame, {args.n_iter} iter bootstrap)", style="cyan")
        t.add_column("Metrica", style="bold")
        t.add_column("PRE [CI95%]", justify="right")
        t.add_column("POST [CI95%]", justify="right")
        t.add_column("Delta [CI95%]  p-value", justify="right")
        t.add_column("Signif.", justify="center")

        report_metrics = {}
        for name, fn in metric_specs:
            cp = bootstrap_ci(outcomes_pre,  fn, n_iter=args.n_iter)
            cq = bootstrap_ci(outcomes_post, fn, n_iter=args.n_iter)
            dd = paired_bootstrap_diff(outcomes_pre, outcomes_post, fn, n_iter=args.n_iter)
            report_metrics[name] = {"pre": cp, "post": cq, "paired_delta": dd}
            t.add_row(
                name,
                f"{cp['point']:.4f} [{cp['low']:.4f}, {cp['high']:.4f}]",
                f"{cq['point']:.4f} [{cq['low']:.4f}, {cq['high']:.4f}]",
                f"{dd['delta']:+.4f} [{dd['low']:+.4f}, {dd['high']:+.4f}]  p={dd['p_value']:.4f}",
                "[green]SI[/green]" if dd["significant"] else "[red]no[/red]",
            )
        console.print(t)

        # DANNO COLLATERALE — media di conteggi per frame, non indicatore
        # binario: usa tutta l'informazione disponibile sugli stessi 80
        # frame positivi (piu' potenza statistica, zero frame aggiuntivi).
        def _collateral_view(outcomes):
            positivi = [o for o in outcomes if o["tp"] + o["fn"] == 1]
            return [
                {"tp": 0, "fn": 0, "fp": o["collateral_count"], "tn": 1}
                for o in positivi
            ]

        collateral_pre  = _collateral_view(outcomes_pre)
        collateral_post = _collateral_view(outcomes_post)
        # media per frame = somma dei conteggi / numero di frame (tn=1 costante
        # per ogni frame -> sum(tn) = n sempre, quindi fp/tn = media campionaria)
        collateral_metric_fn = lambda tp, fp, tn, fn: fp / tn if tn else 0.0

        cp_col = bootstrap_ci(collateral_pre,  collateral_metric_fn, n_iter=args.n_iter)
        cq_col = bootstrap_ci(collateral_post, collateral_metric_fn, n_iter=args.n_iter)
        dd_col = paired_bootstrap_diff(collateral_pre, collateral_post, collateral_metric_fn, n_iter=args.n_iter)

        t2 = Table(title=f"Danno collaterale — media allucinazioni/frame, solo positivi "
                         f"(n={len(collateral_pre)}, conf_threshold={args.conf_threshold})", style="yellow")
        t2.add_column("Metrica", style="bold")
        t2.add_column("PRE [CI95%]", justify="right")
        t2.add_column("POST [CI95%]", justify="right")
        t2.add_column("Delta [CI95%]  p-value", justify="right")
        t2.add_column("Signif.", justify="center")
        t2.add_row(
            "Media collateral/frame",
            f"{cp_col['point']:.4f} [{cp_col['low']:.4f}, {cp_col['high']:.4f}]",
            f"{cq_col['point']:.4f} [{cq_col['low']:.4f}, {cq_col['high']:.4f}]",
            f"{dd_col['delta']:+.4f} [{dd_col['low']:+.4f}, {dd_col['high']:+.4f}]  p={dd_col['p_value']:.4f}",
            "[green]SI[/green]" if dd_col["significant"] else "[red]no[/red]",
        )
        console.print(t2)

        report_metrics["Collateral (media/frame)"] = {"pre": cp_col, "post": cq_col, "paired_delta": dd_col}
        
        suffix = "visdrone" if args.loader == "visdrone" else f"okutama_{args.img_size}"
        if args.loader == "okutama" and args.stride > 1:
            suffix += f"_stride{args.stride}"
        out_path = f"outputs/metrics/full_report_{suffix}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"n_frame": len(outcomes_pre), "n_iter": args.n_iter,
                       "loader": args.loader,
                       "img_size": args.img_size if args.loader == "okutama" else 640,
                       "stride": args.stride, "metrics": report_metrics}, f, indent=2)
        console.print(f"[green]✓ Report salvato in {out_path}[/green]")
        # Extra opzionali (stratificazione per taglia + grafici K). conf_drop
        # RIMOSSO: il relatore ha chiesto il prima/dopo (evasion) significativo,
        # non il calo di confidenza continuo.
        if args.full_report:
            import subprocess
            console.print("\n[bold cyan]Extra: stratificazione per taglia, grafici K[/bold cyan]")
            for cmd in (
                [sys.executable, "tools/stratify_by_size.py", "--data", args.data, "--patch", args.patch] +
                (["--loader", "okutama"] if args.loader == "okutama" else []),
                [sys.executable, "tools/plot_k_selection.py", "--data", args.data, "--max-samples", "300"] +
                (["--loader", "okutama", "--img-size", str(args.img_size)] if args.loader == "okutama" else []),
            ):
                console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
                subprocess.run(cmd, check=False)
        return
    
    # FASE 0: Addestramento Universale
    if args.train_patch:
        console.print("\n[bold green]Avvio Addestramento Universal Patch (SOTA Architecture)[/bold green]")
        try:
            from visdrone_loader import VisDroneLoader
            from okutama_loader import OkutamaLoader
            from patch_optimizer import PatchOptimizer
            from config import CHECKPOINT_FILE
        except ImportError as e:
            console.print(f"[red]Errore dipendenze: {e}[/red]")
            return

        if args.fresh and os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            console.print(f"[yellow]--fresh: rimosso checkpoint precedente ({CHECKPOINT_FILE})[/yellow]")

        if args.loader == "visdrone":
            loader = VisDroneLoader(args.train_dir)
            # TACTICAL FILTER 2026: pre-flight solo per VisDrone — usa API
            # interna (loader._parse_annotation) specifica di VisDroneLoader,
            # non portata su OkutamaLoader.
            from patch_optimizer import tactical_preflight_check
            tactical_preflight_check(loader, low=60.0, high=80.0)
        else:
            loader = OkutamaLoader(args.train_dir, img_size=args.img_size)
            console.print(
                "[dim]Pre-flight Okutama già fatto a mano con "
                "count_negative_candidates.py / stratify_by_size.py "
                "(87.6% frame positivi, vedi sessione precedente) — skip "
                "tactical_preflight_check (API VisDrone-specifica).[/dim]"
            )

        optimizer = PatchOptimizer(model_path="yolov8n.pt")
        patch_out = args.patch_out or BEST_PATCH_FILE

        try:
            results = optimizer.optimize_universal(loader=loader, verbose=True)
            torch.save(results["patch"], patch_out)
            console.print(f"\n[bold green]✓ Universal Patch salvata in: {patch_out}[/bold green]")

            import shutil
            from config import METRICS_JSON_FILE
            suffix = "visdrone" if args.loader == "visdrone" else f"okutama_{args.img_size}"
            if os.path.exists(METRICS_JSON_FILE):
                dest = f"outputs/metrics/training_metrics_{suffix}.json"
                shutil.copy(METRICS_JSON_FILE, dest)
                console.print(f"[dim]Copia training curve salvata in: {dest}[/dim]")
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
            from okutama_loader import OkutamaLoader
            from simulator import evaluate_on_dataset
        except ImportError as e:
            console.print(f"[red]Errore importazione PyTorch/YOLO: {e}[/red]")
            return

        LOADERS = {"visdrone": VisDroneLoader, "okutama": OkutamaLoader}
        loader = (LOADERS[args.loader](args.eval_vision, img_size=args.img_size)
                  if args.loader == "okutama" else LOADERS[args.loader](args.eval_vision))
        console.print(f"Loader ({args.loader}): {len(loader)} frame validi.")

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
        metrics, metrics_tactical, tactical_coverage, per_frame_outcomes = evaluate_on_dataset(
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
            "per_frame_outcomes": per_frame_outcomes,        # BOOTSTRAP CI (relatore)
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
