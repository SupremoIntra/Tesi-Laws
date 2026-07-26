"""
Stratifica l'Evasion Rate per altezza del bounding box della persona.

Obiettivo: verificare se il "soffitto strutturale" (~44%, VisDrone) dipende
dalla dimensione del bersaglio (persone piccole/lontane vs vicine), usando
una patch GIA' addestrata — zero costo di training, solo inferenza.

Su Okutama (prima del training): la patch usata e' quella VisDrone, serve
SOLO a popolare i bucket per confermare la distribuzione di scala reale —
l'evasion rate qui non e' interpretabile come efficacia dell'attacco
(la patch non e' stata addestrata su questo dominio). Guardare i TOTALI
per bucket (colonna "Totali" nell'output), non la evasion rate, finche'
non si passa a una patch addestrata su Okutama.

Letteratura di riferimento: Shrestha et al. (2023, VisDrone) e LFRAP
(2025) attaccano solo veicoli su VisDrone, esplicitamente perche' i
pedoni da vista aerea sono "troppo piccoli" per un budget di patch
limitato. Se questo script conferma un salto netto di evasion rate tra
bucket piccoli e grandi (su VisDrone) o una concentrazione nei bucket
alti (su Okutama), e' la controprova quantitativa.

Uso:
    python tools/stratify_by_size.py \
        --data data/visdrone_val \
        --patch outputs/patches/care_kit_patch_universal.pt

    python tools/stratify_by_size.py \
        --data data/okutama_val --loader okutama \
        --patch outputs/patches/patch_visdrone.pt
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
import cv2
from ultralytics import YOLO

from visdrone_loader import VisDroneLoader
from okutama_loader import OkutamaLoader
from patch_optimizer import get_chest_bbox_proportional
from config import PERSON_CLASS_ID, DRONE_ALTITUDE_M, YOLO_MAX_RANGE

LOADERS = {"visdrone": VisDroneLoader, "okutama": OkutamaLoader}

# Bucket di altezza bbox (pixel, sul canvas del rispettivo loader — 640x640
# per VisDrone, 1280x1280 per Okutama: i bucket sono in pixel assoluti, NON
# comparabili 1:1 tra i due dataset senza tenere conto della risoluzione di
# lavoro, vedi nota nella docstring di okutama_loader.py sul fattore di
# stretch). Il primo bucket (60-100px) e' il minimo accettato dal filtro
# anti-downsampling; l'ultimo (150px+) e' un ingaggio ravvicinato, coerente
# con DRONE_ALTITUDE_M=10 del nostro scenario tattico.
BUCKETS = [(60, 100), (100, 150), (150, 99999)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--patch", type=str, required=True)
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.50)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    loader = LOADERS[args.loader](args.data)
    model = YOLO(args.model)

    patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)
    patch_img = (patch_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    indices = list(range(len(loader)))
    if args.max_samples:
        indices = indices[:args.max_samples]

    counts = {b: [0, 0] for b in BUCKETS}

    for idx in indices:
        img_pil, gt_bboxes = loader.get_sample(idx)
        valid = [b for b in gt_bboxes if (b[3] - b[1]) >= 60]
        if not valid:
            continue

        img_cv = np.array(img_pil)
        for bbox in valid:
            height = bbox[3] - bbox[1]
            bucket = next((b for b in BUCKETS if b[0] <= height < b[1]), None)
            if bucket is None:
                continue

            px1, py1, px2, py2 = get_chest_bbox_proportional(bbox, img_cv.shape[1], img_cv.shape[0])
            pw, ph = px2 - px1, py2 - py1
            if pw <= 0 or ph <= 0:
                continue

            img_patched = img_cv.copy()
            resized = cv2.resize(patch_img, (pw, ph), interpolation=cv2.INTER_AREA)
            img_patched[py1:py2, px1:px2] = resized

            results = model(Image.fromarray(img_patched), verbose=False)
            detected = any(
                (r.boxes is not None and (r.boxes.cls == PERSON_CLASS_ID).any() and
                 r.boxes.conf[(r.boxes.cls == PERSON_CLASS_ID)].max() >= args.conf_threshold)
                for r in results if r.boxes is not None and len(r.boxes) > 0
            )

            counts[bucket][1] += 1
            if not detected:
                counts[bucket][0] += 1

    print(f"\nLoader: {args.loader} | dati: {args.data}")
    print(f"{'Bucket altezza (px)':<22} {'Evasi/Totali':<15} {'Evasion Rate':<15}")
    print("-" * 55)
    for b in BUCKETS:
        evasi, tot = counts[b]
        rate = (evasi / tot * 100) if tot > 0 else 0.0
        label = f"{b[0]}-{b[1] if b[1] < 99999 else '+'}"
        print(f"{label:<22} {evasi}/{tot:<13} {rate:.1f}%")

    print(f"\nRiferimento scenario: DRONE_ALTITUDE_M={DRONE_ALTITUDE_M}m, "
          f"YOLO_MAX_RANGE={YOLO_MAX_RANGE}m (ingaggio ravvicinato)")
    if args.loader == "okutama":
        print("[NOTA] Patch non addestrata su Okutama: guarda i TOTALI per "
              "bucket (distribuzione di scala), non la evasion rate.")


if __name__ == "__main__":
    main()
