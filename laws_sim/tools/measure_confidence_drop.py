"""
Misura il drop di confidenza di YOLO sulla classe "person", persona per
persona, confrontando la stessa immagine con e senza patch applicata.

A differenza di F1/Evasion Rate (metriche aggregate frame-level), questo
script produce un numero per ogni singola persona nel dataset: quanto è
calata la confidenza massima assegnata da YOLO a quella persona specifica.

Uso:
    python tools/measure_confidence_drop.py \
        --data data/visdrone_val \
        --patch outputs/patches/care_kit_patch_universal.pt \
        --max-samples 100
"""
import argparse
import os
import sys
import csv

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
from ultralytics import YOLO

from visdrone_loader import VisDroneLoader
from patch_optimizer import get_chest_bbox_proportional
from config import PERSON_CLASS_ID


def get_max_person_conf(results, person_class=PERSON_CLASS_ID):
    """Massima confidenza sulla classe person in un singolo frame (0.0 se nessuna detection)."""
    best = 0.0
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        mask = (r.boxes.cls == person_class)
        if mask.any():
            best = max(best, float(r.boxes.conf[mask].max()))
    return best


def main():
    parser = argparse.ArgumentParser(description="Confidence drop persona-per-persona, con/senza patch")
    parser.add_argument("--data", type=str, required=True, help="Cartella VisDrone (immagini + annotazioni)")
    parser.add_argument("--patch", type=str, required=True, help="File .pt della patch addestrata")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-csv", type=str, default="outputs/metrics/confidence_drop.csv")
    args = parser.parse_args()

    loader = VisDroneLoader(args.data)
    model = YOLO(args.model)

    patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)
    patch_img = (patch_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    indices = list(range(len(loader)))
    if args.max_samples:
        indices = indices[:args.max_samples]

    rows = []
    for idx in indices:
        img_pil, gt_bboxes = loader.get_sample(idx)
        valid = [b for b in gt_bboxes if (b[3] - b[1]) >= 60]
        if not valid:
            continue

        img_clean = np.array(img_pil)
        img_patched = img_clean.copy()

        for bbox in valid:
            px1, py1, px2, py2 = get_chest_bbox_proportional(bbox, img_clean.shape[1], img_clean.shape[0])
            pw, ph = px2 - px1, py2 - py1
            if pw <= 0 or ph <= 0:
                continue
            import cv2
            resized = cv2.resize(patch_img, (pw, ph), interpolation=cv2.INTER_AREA)
            img_patched[py1:py2, px1:px2] = resized

        conf_clean = get_max_person_conf(model(Image.fromarray(img_clean), verbose=False))
        conf_patched = get_max_person_conf(model(Image.fromarray(img_patched), verbose=False))

        rows.append({
            "frame_idx": idx,
            "n_persons_gt": len(valid),
            "conf_clean": round(conf_clean, 4),
            "conf_patched": round(conf_patched, 4),
            "drop": round(conf_clean - conf_patched, 4),
        })
        print(f"  frame {idx}: conf_clean={conf_clean:.3f} -> conf_patched={conf_patched:.3f} (drop={conf_clean - conf_patched:.3f})")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "n_persons_gt", "conf_clean", "conf_patched", "drop"])
        writer.writeheader()
        writer.writerows(rows)

    drops = [r["drop"] for r in rows]
    print(f"\nFrame analizzati: {len(rows)}")
    print(f"Drop medio: {np.mean(drops):.4f}")
    print(f"Drop mediano: {np.median(drops):.4f}")
    print(f"Frame con drop > 0.3: {sum(1 for d in drops if d > 0.3)}/{len(rows)}")
    print(f"CSV salvato in: {args.out_csv}")


if __name__ == "__main__":
    main()
