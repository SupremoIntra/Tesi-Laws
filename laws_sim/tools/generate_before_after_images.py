"""
Genera immagini di confronto PRIMA/DOPO patch, con bounding box disegnate,
per le slide della presentazione. Seleziona automaticamente i migliori
esempi (persona rilevata prima, evasa dopo, con confidenza ben visibile).

Uso:
    python tools/generate_before_after_images.py \
        --data data/visdrone_val \
        --patch outputs/patches/care_kit_patch_universal.pt \
        --n-examples 4
"""
import argparse
import os
import sys

import numpy as np
import cv2
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
from ultralytics import YOLO

from visdrone_loader import VisDroneLoader
from patch_optimizer import get_chest_bbox_proportional
from config import PERSON_CLASS_ID


def get_target_conf(results, target_bbox, person_class=PERSON_CLASS_ID):
    """
    Confidenza del box che corrisponde al bersaglio noto (target_bbox), non il
    massimo su tutta l'immagine. Sceglie il box person-class con maggiore IoU
    (o overlap del centro) rispetto a target_bbox, per non confondere il
    bersaglio con altre persone eventualmente presenti nella stessa scena.
    """
    tx1, ty1, tx2, ty2 = target_bbox
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    best_conf = 0.0
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        mask = (r.boxes.cls == person_class)
        for box, conf in zip(r.boxes.xyxy[mask], r.boxes.conf[mask]):
            x1, y1, x2, y2 = [float(v) for v in box]
            # Il box "appartiene" al bersaglio se il centro del bersaglio
            # cade dentro il box rilevato (o viceversa) - robusto a piccoli
            # spostamenti della box stimata sotto attacco.
            if (x1 <= tcx <= x2 and y1 <= tcy <= y2):
                best_conf = max(best_conf, float(conf))
    return best_conf


def draw_detections(img: np.ndarray, results, color=(0, 255, 0)) -> np.ndarray:
    """Disegna le box rilevate da YOLO per la classe person, con la confidenza."""
    out = img.copy()
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        mask = (r.boxes.cls == PERSON_CLASS_ID)
        for box, conf in zip(r.boxes.xyxy[mask], r.boxes.conf[mask]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"person {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--patch", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.50)
    parser.add_argument("--out-dir", type=str, default="outputs/metrics/before_after")
    parser.add_argument("--max-samples", type=int, default=14210)
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone")
    parser.add_argument("--img-size", type=int, default=960)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from okutama_loader import OkutamaLoader
    LOADERS = {"visdrone": VisDroneLoader, "okutama": OkutamaLoader}
    loader = (LOADERS[args.loader](args.data, img_size=args.img_size)
              if args.loader == "okutama" else LOADERS[args.loader](args.data))
    model = YOLO(args.model)

    patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)
    patch_img = (patch_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    indices = list(range(len(loader)))[:args.max_samples]
    candidates = []  # (drop_conf, idx, bbox_area)

    print(f"Scansione di {len(indices)} frame in cerca di buoni esempi...")
    for idx in indices:
        img_pil, gt_bboxes = loader.get_sample(idx)
        valid = [b for b in gt_bboxes if (b[3] - b[1]) >= 60]
        if len(valid) != 1:
            continue  # frame con una sola persona: piu' chiaro da mostrare in slide

        bbox = valid[0]
        img_clean = np.array(img_pil)

        res_clean = model(Image.fromarray(img_clean), verbose=False)
        conf_clean = get_target_conf(res_clean, bbox)
        if conf_clean < args.conf_threshold:
            continue  # deve essere rilevata PRIMA della patch

        px1, py1, px2, py2 = get_chest_bbox_proportional(bbox, img_clean.shape[1], img_clean.shape[0])
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            continue

        img_patched = img_clean.copy()
        resized = cv2.resize(patch_img, (pw, ph), interpolation=cv2.INTER_AREA)
        img_patched[py1:py2, px1:px2] = resized

        res_patched = model(Image.fromarray(img_patched), verbose=False)
        conf_patched = get_target_conf(res_patched, bbox)

        drop = conf_clean - conf_patched
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        candidates.append((drop, conf_patched < args.conf_threshold, idx, area, conf_clean, conf_patched))
        print(f"  frame {idx}: conf {conf_clean:.2f} -> {conf_patched:.2f} (drop={drop:.2f})")

    # Preferiamo: evasione completa (conf_patched < soglia), poi il drop maggiore, poi bbox piu' grande (piu' leggibile in slide)
    candidates.sort(key=lambda c: (c[1], c[0], c[3]), reverse=True)
    best = candidates[:args.n_examples]

    if not best:
        print("\nNessun buon esempio trovato nei frame scansionati. Prova ad aumentare --max-samples.")
        return

    print(f"\nGenero {len(best)} immagini di confronto in {args.out_dir}/...")
    for rank, (drop, evaded, idx, area, conf_clean, conf_patched) in enumerate(best):
        img_pil, gt_bboxes = loader.get_sample(idx)
        bbox = [b for b in gt_bboxes if (b[3] - b[1]) >= 60][0]
        img_clean = np.array(img_pil)

        px1, py1, px2, py2 = get_chest_bbox_proportional(bbox, img_clean.shape[1], img_clean.shape[0])
        pw, ph = px2 - px1, py2 - py1
        img_patched = img_clean.copy()
        resized = cv2.resize(patch_img, (pw, ph), interpolation=cv2.INTER_AREA)
        img_patched[py1:py2, px1:px2] = resized

        res_clean = model(Image.fromarray(img_clean), verbose=False)
        res_patched = model(Image.fromarray(img_patched), verbose=False)

        img_clean_annotated = draw_detections(img_clean, res_clean, color=(0, 200, 0))
        img_patched_annotated = draw_detections(img_patched, res_patched, color=(0, 0, 255))

        # Crop centrato sulla persona con un margine, per leggibilita' in slide
        h, w = img_clean.shape[:2]
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        half = max(150, (bbox[2] - bbox[0]), (bbox[3] - bbox[1])) 
        x1c, y1c = max(0, cx - half), max(0, cy - half)
        x2c, y2c = min(w, cx + half), min(h, cy + half)

        crop_clean = img_clean_annotated[y1c:y2c, x1c:x2c]
        crop_patched = img_patched_annotated[y1c:y2c, x1c:x2c]

        # Ridimensiona entrambi alla stessa altezza e affianca
        target_h = 500
        def resize_to_h(im, th):
            r = th / im.shape[0]
            return cv2.resize(im, (int(im.shape[1] * r), th))
        crop_clean = resize_to_h(crop_clean, target_h)
        crop_patched = resize_to_h(crop_patched, target_h)

        gap = np.ones((target_h, 8, 3), dtype=np.uint8) * 255
        combined = np.hstack([crop_clean, gap, crop_patched])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(args.out_dir, f"example_{rank+1}_conf_{conf_clean:.2f}_to_{conf_patched:.2f}.png")
        cv2.imwrite(out_path, combined_bgr)
        print(f"  Salvato: {out_path} (evasione completa: {evaded})")

    print("\nFatto. Scegli le 2-3 immagini migliori (evasione completa, drop piu' alto) per il PPT.")


if __name__ == "__main__":
    main()
