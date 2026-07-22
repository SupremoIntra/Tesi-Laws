"""
Confronto PRE/POST attacco con Confidence Interval bootstrap (metodo dei
percentili, richiesta esplicita del relatore).

Esegue evaluate_on_dataset DUE volte sullo stesso valset:
    PRE  = senza patch (comportamento naturale di YOLO)
    POST = con la patch addestrata (sotto attacco)

Per ciascuna, calcola F1 e la media geometrica sqrt(sensitivity*specificity)
con intervallo di confidenza al 95% (bootstrap percentile, 10.000
iterazioni di default) e stampa un verdetto esplicito di significatività:
se gli intervalli PRE e POST si sovrappongono, la differenza potrebbe
NON essere statisticamente significativa.

Uso:
    python tools/bootstrap_ci_report.py \
        --data data/visdrone_val \
        --patch outputs/patches/care_kit_patch_universal.pt \
        --n-iter 10000
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch

from visdrone_loader import VisDroneLoader
from simulator import evaluate_on_dataset
from metrics import bootstrap_ci, geometric_mean_recall
from metrics import f1_from_counts, gmean_from_counts


def report_ci(label: str, outcomes, n_iter: int):
    ci_f1 = bootstrap_ci(outcomes, f1_from_counts, n_iter=n_iter)
    ci_gm = bootstrap_ci(outcomes, gmean_from_counts, n_iter=n_iter)
    print(f"\n[{label}] n_frame={len(outcomes)}")
    print(f"  F1                    = {ci_f1['point']:.4f}  CI95%=[{ci_f1['low']:.4f}, {ci_f1['high']:.4f}]")
    print(f"  sqrt(sens*spec)       = {ci_gm['point']:.4f}  CI95%=[{ci_gm['low']:.4f}, {ci_gm['high']:.4f}]")
    return ci_f1, ci_gm


def overlap(ci_a, ci_b) -> bool:
    return ci_a["low"] <= ci_b["high"] and ci_b["low"] <= ci_a["high"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--patch", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--out", type=str, default="outputs/metrics/bootstrap_ci_report.json")
    args = parser.parse_args()

    loader = VisDroneLoader(args.data)
    patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)

    print("=" * 70)
    print("PRE-attacco (nessuna patch, comportamento naturale di YOLO)")
    print("=" * 70)
    _, _, _, outcomes_pre = evaluate_on_dataset(
        loader=loader, patch_tensor=None, model_path=args.model,
        max_samples=args.max_samples, verbose=False
    )
    ci_f1_pre, ci_gm_pre = report_ci("PRE", outcomes_pre, args.n_iter)

    print("\n" + "=" * 70)
    print("POST-attacco (con patch)")
    print("=" * 70)
    _, _, _, outcomes_post = evaluate_on_dataset(
        loader=loader, patch_tensor=patch_tensor, model_path=args.model,
        max_samples=args.max_samples, verbose=False
    )
    ci_f1_post, ci_gm_post = report_ci("POST", outcomes_post, args.n_iter)

    f1_overlap = overlap(ci_f1_pre, ci_f1_post)
    gm_overlap = overlap(ci_gm_pre, ci_gm_post)

    from metrics import paired_bootstrap_diff
    res_f1 = paired_bootstrap_diff(outcomes_pre, outcomes_post, f1_from_counts, n_iter=args.n_iter)
    res_gm = paired_bootstrap_diff(outcomes_pre, outcomes_post, gmean_from_counts, n_iter=args.n_iter)
    
    
    from metrics import paired_bootstrap_diff
    res_f1 = paired_bootstrap_diff(outcomes_pre, outcomes_post, f1_from_counts, n_iter=args.n_iter)
    res_gm = paired_bootstrap_diff(outcomes_pre, outcomes_post, gmean_from_counts, n_iter=args.n_iter)

    def _fmt(r):
        return (f"delta={r['delta']:+.4f}  CI95%=[{r['low']:+.4f}, {r['high']:+.4f}]  "
                f"p={r['p_value']:.4f}  ->  {'SIGNIFICATIVO' if r['significant'] else 'NON significativo'}")

    print("\n" + "=" * 70)
    print("VERDETTO DI SIGNIFICATIVITA' (95%)")
    print("=" * 70)
    print("[Metodo 1 - CI indipendenti PRE vs POST (conservativo)]")
    print(f"  F1:              {'SOVRAPPOSTI -> non dimostrato' if f1_overlap else 'DISGIUNTI -> significativo'}")
    print(f"  sqrt(sens*spec): {'SOVRAPPOSTI -> non dimostrato' if gm_overlap else 'DISGIUNTI -> significativo'}")
    print("[Metodo 2 - bootstrap APPAIATO del delta (piu' potente, usa la corrispondenza per-frame)]")
    print(f"  F1:              {_fmt(res_f1)}")
    print(f"  sqrt(sens*spec): {_fmt(res_gm)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "n_iter": args.n_iter,
        "pre":  {"f1": ci_f1_pre,  "geometric_mean": ci_gm_pre,  "n_frame": len(outcomes_pre)},
        "post": {"f1": ci_f1_post, "geometric_mean": ci_gm_post, "n_frame": len(outcomes_post)},
        "ci_indipendenti_significativo_95": {"f1": not f1_overlap, "geometric_mean": not gm_overlap},
        "paired_delta": {"f1": res_f1, "geometric_mean": res_gm},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport completo salvato in: {args.out}")

if __name__ == "__main__":
    main()
