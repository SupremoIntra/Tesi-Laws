"""Genera coppie di immagini PRIMA/DOPO per la figura di tesi e per le slide.

Revisione della versione precedente. Tre difetti della v1 sono corretti qui, tutti
verificati sulle otto immagini gia' prodotte:

D1  ``draw_detections`` disegnava un rettangolo pieno sotto ogni etichetta. Con due
    rilevamenti sovrapposti la seconda etichetta cancellava la prima, per cui
    ``example_1_conf_0_74_to_0_00.png`` mostra a schermo ``0.26``. Qui l'etichetta
    e' disegnata **solo per il box appaiato al bersaglio**, e le altre detection
    ricevono il solo contorno.

D2  L'annotazione avveniva sull'immagine intera e il ritaglio arrivava dopo, per cui
    un box piu' grande della finestra di ritaglio spariva del tutto:
    ``example_3_conf_0_71_to_0_35.png`` ha ``conf_patched = 0.35`` e nessun box
    visibile. Qui si ritaglia prima, si trasla in coordinate del ritaglio e si
    disegna dopo, con clipping esplicito.

D3  Il filtro ``len(valid) != 1`` scartava ogni fotogramma multi-bersaglio, per cui
    l'unico esempio di specificita' disponibile ha come seconda persona un bersaglio
    **sotto** la soglia di validita' di 60 px. La modalita' ``--mode specificity``
    seleziona fotogrammi con due o piu' bersagli **validi**, applica la patch a uno
    solo e verifica che gli altri restino rilevati: e' l'evidenza visiva che il
    Capitolo 5 afferma oggi soltanto a parole.

L'architettura invariante non e' toccata: ``get_chest_bbox_proportional`` e' importata
e usata come sorgente unica della collocazione della patch.

Uso::

    python tools/generate_before_after_images.py \\
        --data data/visdrone_val --loader visdrone \\
        --patch outputs/patches/care_kit_patch_universal.pt \\
        --mode evasion --n-examples 6

    python tools/generate_before_after_images.py \\
        --data data/okutama_val --loader okutama --img-size 960 \\
        --patch outputs/patches/care_kit_patch_universal.pt \\
        --mode specificity --n-examples 4
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Final, Literal, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config import PERSON_CLASS_ID  # noqa: E402
from patch_optimizer import get_chest_bbox_proportional  # noqa: E402
from visdrone_loader import VisDroneLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("before_after")

BBox = tuple[float, float, float, float]

#: Soglia di validita' tattica. Deve coincidere con il filtro di rilevanza usato in
#: fase di ottimizzazione (Sezione 4.3): un bersaglio piu' basso non contribuisce al
#: gradiente e non deve comparire come esempio.
MIN_TARGET_HEIGHT_PX: Final[int] = 60

COLOR_CONTROL: Final[tuple[int, int, int]] = (0, 200, 0)
COLOR_ATTACK: Final[tuple[int, int, int]] = (0, 0, 255)
COLOR_BYSTANDER: Final[tuple[int, int, int]] = (255, 170, 0)


@dataclass(frozen=True)
class Candidate:
    """Un fotogramma selezionato, con le misure che ne motivano la scelta.

    Attributes:
        idx: Indice nel loader.
        target: Riquadro annotato su cui la patch viene composta.
        others: Riquadri annotati validi non perturbati, vuoto in modalita' evasion.
        conf_clean: Confidenza appaiata al bersaglio in condizione di controllo.
        conf_patched: Confidenza appaiata al bersaglio sotto attacco.
        others_kept: Numero di bystander ancora rilevati sopra soglia sotto attacco.
        evaded: Vero se il bersaglio scende sotto la soglia operativa.
    """

    idx: int
    target: BBox
    others: tuple[BBox, ...]
    conf_clean: float
    conf_patched: float
    others_kept: int
    evaded: bool

    @property
    def drop(self) -> float:
        """Caduta di confidenza sul solo bersaglio appaiato."""
        return self.conf_clean - self.conf_patched

    @property
    def height_px(self) -> float:
        """Altezza del bersaglio, che ne determina il bucket di stratificazione."""
        return self.target[3] - self.target[1]


def matched_confidence(results: Sequence, target: BBox) -> tuple[float, BBox | None]:
    """Confidenza del rilevamento appaiato a un bersaglio noto.

    Il criterio e' il contenimento del centro del bersaglio nel box rilevato, come
    nella v1, per restare robusti allo spostamento del box stimato sotto attacco. A
    differenza della v1 restituisce anche la geometria del box vincente, necessaria
    per etichettare quel box e soltanto quello (difetto D1).

    Args:
        results: Uscita di ``YOLO.__call__`` sul fotogramma.
        target: Riquadro annotato del bersaglio, in coordinate immagine.

    Returns:
        Coppia (confidenza, riquadro rilevato). ``(0.0, None)`` se nessun box appaia.
    """
    tx1, ty1, tx2, ty2 = target
    tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    best_conf, best_box = 0.0, None
    for res in results:
        if res.boxes is None or len(res.boxes) == 0:
            continue
        keep = res.boxes.cls == PERSON_CLASS_ID
        for box, conf in zip(res.boxes.xyxy[keep], res.boxes.conf[keep]):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x1 <= tcx <= x2 and y1 <= tcy <= y2 and float(conf) > best_conf:
                best_conf, best_box = float(conf), (x1, y1, x2, y2)
    return best_conf, best_box


def compose_patch(image: np.ndarray, target: BBox, patch_rgb: np.ndarray) -> np.ndarray:
    """Compone la patch sul torace del bersaglio, in proporzione al riquadro.

    Args:
        image: Fotogramma RGB non perturbato.
        target: Riquadro annotato del bersaglio.
        patch_rgb: Patch gia' convertita in uint8 RGB.

    Returns:
        Copia del fotogramma con la patch composta. L'originale non e' modificato.
    """
    px1, py1, px2, py2 = get_chest_bbox_proportional(target, image.shape[1], image.shape[0])
    width, height = px2 - px1, py2 - py1
    if width <= 0 or height <= 0:
        return image.copy()
    out = image.copy()
    out[py1:py2, px1:px2] = cv2.resize(patch_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return out


def crop_window(image_shape: tuple[int, ...], target: BBox, margin: int) -> tuple[int, int, int, int]:
    """Finestra quadrata centrata sul bersaglio, con margine minimo garantito.

    Args:
        image_shape: Forma del fotogramma, ``(h, w, c)``.
        target: Riquadro del bersaglio.
        margin: Semilato minimo in pixel.

    Returns:
        Tupla ``(x1, y1, x2, y2)`` gia' vincolata ai bordi dell'immagine.
    """
    height, width = image_shape[:2]
    cx = int((target[0] + target[2]) / 2)
    cy = int((target[1] + target[3]) / 2)
    half = int(max(margin, target[2] - target[0], target[3] - target[1]))
    return max(0, cx - half), max(0, cy - half), min(width, cx + half), min(height, cy + half)


def annotate_in_crop(
    crop: np.ndarray,
    window: tuple[int, int, int, int],
    boxes: Sequence[tuple[BBox, float]],
    labelled: BBox | None,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Disegna i rilevamenti in coordinate del ritaglio, etichettando un solo box.

    Traslare prima di disegnare risolve D2: un box piu' grande della finestra resta
    visibile perche' ``cv2.rectangle`` lo clippa ai bordi del ritaglio invece di
    finire fuori tela. Etichettare il solo box appaiato risolve D1: nessuna etichetta
    puo' piu' coprirne un'altra.

    Args:
        crop: Ritaglio RGB su cui disegnare.
        window: Finestra ``(x1, y1, x2, y2)`` usata per il ritaglio.
        boxes: Rilevamenti ``(riquadro, confidenza)`` in coordinate immagine.
        labelled: Riquadro da etichettare, tipicamente quello appaiato al bersaglio.
        color: Colore del contorno del box etichettato.

    Returns:
        Copia annotata del ritaglio.
    """
    out = crop.copy()
    ox, oy = window[0], window[1]
    crop_h, crop_w = out.shape[:2]

    for box, conf in boxes:
        x1, y1 = int(box[0] - ox), int(box[1] - oy)
        x2, y2 = int(box[2] - ox), int(box[3] - oy)
        if x2 < 0 or y2 < 0 or x1 > crop_w or y1 > crop_h:
            continue  # interamente fuori dal ritaglio
        is_target = labelled is not None and np.allclose(box, labelled, atol=1e-6)
        stroke = color if is_target else COLOR_BYSTANDER
        cv2.rectangle(out, (x1, y1), (x2, y2), stroke, 2 if is_target else 1)
        if not is_target:
            continue

        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # L'etichetta va sopra il box, ma rientra sotto il bordo superiore se non
        # c'e' spazio: e' il caso in cui la v1 la perdeva del tutto.
        ly = y1 - 4 if y1 - th - 8 >= 0 else min(crop_h - 4, y2 + th + 4)
        lx = int(np.clip(x1, 0, max(0, crop_w - tw - 6)))
        cv2.rectangle(out, (lx, ly - th - 6), (lx + tw + 6, ly + 2), stroke, -1)
        cv2.putText(out, label, (lx + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


def scan(
    loader,
    model: YOLO,
    patch_rgb: np.ndarray,
    mode: Literal["evasion", "specificity"],
    conf_threshold: float,
    max_samples: int,
) -> list[Candidate]:
    """Scandisce l'insieme di validazione e raccoglie i fotogrammi utilizzabili.

    In modalita' ``evasion`` si richiede un unico bersaglio valido, come nella v1. In
    modalita' ``specificity`` se ne richiedono almeno due: la patch e' composta solo
    sul primo e si conta quanti degli altri restano rilevati sopra soglia.

    Args:
        loader: Loader del dataset, con ``get_sample`` e ``__len__``.
        model: Rilevatore YOLO gia' istanziato.
        patch_rgb: Patch in uint8 RGB.
        mode: Criterio di selezione.
        conf_threshold: Soglia operativa, coerente con quella delle metriche in tesi.
        max_samples: Limite superiore di fotogrammi scanditi.

    Returns:
        Lista di candidati, non ordinata.
    """
    found: list[Candidate] = []
    n_scan = min(len(loader), max_samples)
    _LOG.info("Scansione di %d fotogrammi in modalita' %s...", n_scan, mode)

    for idx in range(n_scan):
        try:
            img_pil, gt_boxes = loader.get_sample(idx)
        except Exception:  # loader difettoso su un singolo file: non deve fermare il run
            _LOG.warning("Fotogramma %d illeggibile, saltato.", idx, exc_info=True)
            continue

        valid = [tuple(b) for b in gt_boxes if (b[3] - b[1]) >= MIN_TARGET_HEIGHT_PX]
        if mode == "evasion" and len(valid) != 1:
            continue
        if mode == "specificity" and len(valid) < 2:
            continue

        # Il bersaglio e' il piu' alto: e' quello su cui la patch ha piu' superficie.
        valid.sort(key=lambda b: b[3] - b[1], reverse=True)
        target, others = valid[0], tuple(valid[1:])
        clean = np.array(img_pil)

        res_clean = model(Image.fromarray(clean), verbose=False)
        conf_clean, _ = matched_confidence(res_clean, target)
        if conf_clean < conf_threshold:
            continue  # deve essere rilevato PRIMA, altrimenti non c'e' nulla da mostrare
        if mode == "specificity" and not all(
            matched_confidence(res_clean, o)[0] >= conf_threshold for o in others
        ):
            continue  # i bystander devono partire rilevati, o la specificita' non si vede

        patched = compose_patch(clean, target, patch_rgb)
        res_patched = model(Image.fromarray(patched), verbose=False)
        conf_patched, _ = matched_confidence(res_patched, target)
        kept = sum(
            1 for o in others if matched_confidence(res_patched, o)[0] >= conf_threshold
        )

        found.append(
            Candidate(
                idx=idx,
                target=target,
                others=others,
                conf_clean=conf_clean,
                conf_patched=conf_patched,
                others_kept=kept,
                evaded=conf_patched < conf_threshold,
            )
        )
        _LOG.info(
            "  frame %d: %.2f -> %.2f (h=%.0f px, bystander rilevati %d/%d)",
            idx, conf_clean, conf_patched, target[3] - target[1], kept, len(others),
        )
        del clean, patched, res_clean, res_patched
        if idx % 200 == 0:
            gc.collect()

    return found


def render(cand: Candidate, loader, model: YOLO, patch_rgb: np.ndarray,
           margin: int, target_height: int) -> np.ndarray:
    """Produce il pannello affiancato controllo / attacco per un candidato.

    Args:
        cand: Candidato selezionato.
        loader: Loader del dataset.
        model: Rilevatore YOLO.
        patch_rgb: Patch in uint8 RGB.
        margin: Semilato minimo del ritaglio.
        target_height: Altezza in pixel a cui normalizzare i due pannelli.

    Returns:
        Immagine BGR pronta per ``cv2.imwrite``.
    """
    img_pil, _ = loader.get_sample(cand.idx)
    clean = np.array(img_pil)
    patched = compose_patch(clean, cand.target, patch_rgb)

    res_clean = model(Image.fromarray(clean), verbose=False)
    res_patched = model(Image.fromarray(patched), verbose=False)
    _, box_clean = matched_confidence(res_clean, cand.target)
    _, box_patched = matched_confidence(res_patched, cand.target)

    def detections(results: Sequence) -> list[tuple[BBox, float]]:
        out: list[tuple[BBox, float]] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            keep = res.boxes.cls == PERSON_CLASS_ID
            for box, conf in zip(res.boxes.xyxy[keep], res.boxes.conf[keep]):
                out.append((tuple(float(v) for v in box), float(conf)))
        return out

    window = crop_window(clean.shape, cand.target, margin)
    x1, y1, x2, y2 = window
    left = annotate_in_crop(clean[y1:y2, x1:x2], window, detections(res_clean),
                            box_clean, COLOR_CONTROL)
    right = annotate_in_crop(patched[y1:y2, x1:x2], window, detections(res_patched),
                             box_patched, COLOR_ATTACK)

    def to_height(img: np.ndarray) -> np.ndarray:
        ratio = target_height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), target_height))

    gap = np.full((target_height, 8, 3), 255, dtype=np.uint8)
    combined = np.hstack([to_height(left), gap, to_height(right)])
    del clean, patched, res_clean, res_patched
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)


def main() -> None:
    """Punto di ingresso."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--patch", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--loader", choices=["visdrone", "okutama"], default="visdrone")
    parser.add_argument("--mode", choices=["evasion", "specificity"], default="evasion")
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--conf-threshold", type=float, default=0.50,
                        help="Soglia operativa. Deve coincidere con quella delle metriche in tesi.")
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--max-samples", type=int, default=14210)
    parser.add_argument("--crop-margin", type=int, default=150)
    parser.add_argument("--min-index-gap", type=int, default=300,
                        help="Distanza minima in fotogrammi fra due esempi selezionati. "
                             "A 30 fps, 300 fotogrammi sono dieci secondi: sotto quella "
                             "soglia i candidati appartengono alla stessa scena e le "
                             "immagini risultano quasi identiche.")
    parser.add_argument("--panel-height", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="outputs/metrics/before_after")
    parser.add_argument("--dump-patch", action="store_true",
                        help="Salva anche la patch da sola: la tesi non la mostra mai.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from okutama_loader import OkutamaLoader  # import tardivo: dipendenza opzionale
    loader = (OkutamaLoader(args.data, img_size=args.img_size)
              if args.loader == "okutama" else VisDroneLoader(args.data))
    model = YOLO(args.model)

    patch_tensor = torch.load(args.patch, map_location="cpu", weights_only=True)
    patch_rgb = (patch_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    del patch_tensor
    gc.collect()

    if args.dump_patch:
        patch_path = os.path.join(args.out_dir, "patch_universale.png")
        cv2.imwrite(patch_path, cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
        _LOG.info("Patch salvata in %s", patch_path)

    candidates = scan(loader, model, patch_rgb, args.mode, args.conf_threshold,
                      args.max_samples)
    if not candidates:
        _LOG.error("Nessun candidato. Aumentare --max-samples o abbassare --conf-threshold.")
        return

    if args.mode == "specificity":
        # Il caso forte e' bersaglio evaso con TUTTI i bystander ancora rilevati.
        candidates.sort(key=lambda c: (c.evaded, c.others_kept, c.drop), reverse=True)
    else:
        candidates.sort(key=lambda c: (c.evaded, c.drop, c.height_px), reverse=True)

    # Un video a 30 fps produce candidati consecutivi visivamente indistinguibili:
    # la selezione degli esempi richiede la stessa decorrelazione applicata al
    # campionamento statistico (Sezione 4.6), qui su scala di scena anziche' di
    # indipendenza fra osservazioni. Senza questo filtro i primi quattro candidati
    # su Okutama cadono entro cinque fotogrammi l'uno dall'altro.
    selected: list[Candidate] = []
    for cand in candidates:
        if any(abs(cand.idx - other.idx) < args.min_index_gap for other in selected):
            continue
        selected.append(cand)
        if len(selected) == args.n_examples:
            break

    _LOG.info("Genero %d immagini in %s/ (%d candidati, distanza minima %d frame)",
              len(selected), args.out_dir, len(candidates), args.min_index_gap)
    for rank, cand in enumerate(selected, start=1):
        try:
            panel = render(cand, loader, model, patch_rgb, args.crop_margin,
                           args.panel_height)
        except Exception:
            _LOG.warning("Rendering fallito sul fotogramma %d, saltato.", cand.idx,
                         exc_info=True)
            continue
        # Nome auto-descrittivo: dataset, modalita', indice reale, altezza del
        # bersaglio (bucket di stratificazione) e bystander conservati.
        name = (f"{args.loader}_{args.mode}_{rank:02d}_idx{cand.idx}"
                f"_h{int(cand.height_px)}px"
                f"_conf{cand.conf_clean:.2f}to{cand.conf_patched:.2f}"
                f"_keep{cand.others_kept}of{len(cand.others)}.png")
        cv2.imwrite(os.path.join(args.out_dir, name), panel)
        _LOG.info("  %s", name)
        del panel
        gc.collect()

    _LOG.info("Fatto.")


if __name__ == "__main__":
    main()
