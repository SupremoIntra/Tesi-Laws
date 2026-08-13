"""
Stratifica l'Evasion Rate per altezza del bounding box della persona.

Obiettivo: verificare se il degrado dipende dalla dimensione del bersaglio
(persone piccole/lontane vs vicine), usando una patch gia' addestrata -
zero costo di training, solo inferenza.

L'esito e' attribuito per bersaglio mediante corrispondenza IoU (soglia
IOU_IGNORE_THRESHOLD, importata da simulator.py per coerenza con la
convenzione di valutazione usata nel resto della pipeline): un rilevamento
conta come "visto" solo se sovrappone quello specifico bersaglio, non un
soggetto qualunque nel fotogramma. Necessario su dataset con piu' soggetti
per fotogramma (es. Okutama), dove altrimenti un rilevamento su un soggetto
diverso farebbe erroneamente contare il bersaglio come non evaso.

Uso:
    # condizione di controllo (nessuna patch)
    python tools/stratify_by_size.py \\
        --data data/okutama_val --loader okutama --img-size 960 --no-patch

    # condizione sotto attacco
    python tools/stratify_by_size.py \\
        --data data/okutama_val --loader okutama --img-size 960 \\
        --patch outputs/patches/patch_okutama.pt
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
from simulator import iou, IOU_IGNORE_THRESHOLD

LOADERS = {"visdrone": VisDroneLoader, "okutama": OkutamaLoader}

# Bucket di altezza bbox (pixel, sul canvas del rispettivo loader - 640x640
# per VisDrone, 1280x1280 o 960x960 per Okutama a seconda di --img-size: i
# bucket sono in pixel assoluti, NON comparabili 1:1 tra risoluzioni diverse
# senza tenere conto del fattore di stretch). Il primo bucket (60-100px) e'
# il minimo accettato dal filtro anti-downsampling; l'ultimo (150px+) e' un
# ingaggio ravvicinato, coerente con DRONE_ALTITUDE_M.
BUCKETS = [(60, 100), (100, 150), (150, 99999)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--patch", type=str, default=None)
    parser.add_argument("--no-patch", action="store_true",
                         help="Condizione di controllo: nessuna patch composta")
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone")
    parser.add_argument("--img-size", type=int, default=960, help="Canvas per loader okutama")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.50)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not args.no_patch and not args.patch:
        parser.error("serve --patch, oppure --no-patch per la condizione di controllo")

    loader = (LOADERS[args.loader](args.data, img_size=args.img_size)
              if args.loader == "okutama" else LOADERS[args.loader](args.data))
    model = YOLO(args.model)

    patch_img = None
    if not args.no_patch:
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

            if args.no_patch:
                img_patched = img_cv
            else:
                px1, py1, px2, py2 = get_chest_bbox_proportional(
                    bbox, img_cv.shape[1], img_cv.shape[0]
                )
                pw, ph = px2 - px1, py2 - py1
                if pw <= 0 or ph <= 0:
                    continue
                img_patched = img_cv.copy()
                resized = cv2.resize(patch_img, (pw, ph), interpolation=cv2.INTER_AREA)
                img_patched[py1:py2, px1:px2] = resized

            results = model(Image.fromarray(img_patched), verbose=False)

            # Detection IoU-matched: un rilevamento conta solo se sovrappone
            # QUESTO bersaglio (bbox), non un soggetto qualunque nel frame.
            detected = False
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                keep = (r.boxes.cls == PERSON_CLASS_ID) & (r.boxes.conf >= args.conf_threshold)
                for det in r.boxes.xyxy[keep].tolist():
                    if iou(tuple(det), tuple(bbox)) >= IOU_IGNORE_THRESHOLD:
                        detected = True
                        break
                if detected:
                    break

            counts[bucket][1] += 1
            if not detected:
                counts[bucket][0] += 1

    condizione = "CONTROLLO (no patch)" if args.no_patch else f"ATTACCO (patch={args.patch})"
    print(f"\nLoader: {args.loader} | dati: {args.data} | condizione: {condizione}")
    print(f"{'Bucket altezza (px)':<22} {'Evasi/Totali':<15} {'Evasion Rate':<15}")
    print("-" * 55)
    for b in BUCKETS:
        evasi, tot = counts[b]
        rate = (evasi / tot * 100) if tot > 0 else 0.0
        label = f"{b[0]}-{b[1] if b[1] < 99999 else '+'}"
        print(f"{label:<22} {evasi}/{tot:<13} {rate:.1f}%")

    print(f"\nRiferimento scenario: DRONE_ALTITUDE_M={DRONE_ALTITUDE_M}m, "
          f"YOLO_MAX_RANGE={YOLO_MAX_RANGE}m (ingaggio ravvicinato)")


if __name__ == "__main__":
    main()
