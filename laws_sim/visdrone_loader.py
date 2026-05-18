"""
VisDrone DataLoader per LAWS-SIM.

Carica immagini e bounding box reali dal dataset VisDrone2019-DET.

Formato annotazioni VisDrone (una riga per oggetto):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

Categorie rilevanti:
    1 = pedestrian
    2 = people (gruppo)
    (ignoriamo veicoli, categorie 3-10)

Struttura attesa su disco:
    visdrone_root/
    ├── images/     (*.jpg)
    └── annotations/ (*.txt)

Download ufficiale:
    https://github.com/VisDrone/VisDrone-Dataset
    Bastano le immagini di test (VisDrone2019-DET-val) per iniziare:
    ~500 immagini, ~370 MB

Uso base:
    loader = VisDroneLoader("/path/to/visdrone")
    for img_pil, bboxes in loader.iter_batches(batch_size=4):
        # img_pil: lista di PIL.Image (640×640)
        # bboxes: lista di liste [(x1,y1,x2,y2), ...] per ogni immagine
        patch_tensor = apply_patch(patch, img_pil, bboxes)
        loss = yolo_loss(patch_tensor)
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional, Iterator

import numpy as np
from PIL import Image

# Categorie VisDrone che corrispondono a "persona"
# 1=pedestrian, 2=people, 0=ignore region (skip)
PERSON_CATEGORIES = {1, 2}
MIN_BBOX_AREA = 100   # ignora bbox troppo piccoli (noise): < 10×10 pixel


class VisDroneLoader:
    """
    DataLoader per VisDrone2019-DET.

    Costruttore:
        root_dir: cartella con images/ e annotations/
        img_size: dimensione a cui ridimensionare le immagini (default 640, come YOLOv8)
        min_persons: ignora frame senza almeno N persone annotate
        seed: per riproducibilità

    Iterazione:
        loader.iter_batches(batch_size=4)   → yields (imgs, bboxes_list)
        loader.get_sample(idx)              → singola immagine con bboxes

    Nota sul ridimensionamento:
        VisDrone ha immagini originali a ~1920×1080. Le ridimensioniamo a
        IMG_SIZE×IMG_SIZE (640×640). I bbox vengono scalati di conseguenza.
        Questo è lo stesso preprocessing di YOLOv8.
    """

    def __init__(self, root_dir: str, img_size: int = 640,
                 min_persons: int = 1, seed: int = 42):
        self.root     = Path(root_dir)
        self.img_size = img_size
        self.seed     = seed
        random.seed(seed)

        self.img_dir  = self.root / "images"
        self.ann_dir  = self.root / "annotations"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Directory immagini non trovata: {self.img_dir}\n"
                f"Struttura attesa: {self.root}/images/ e {self.root}/annotations/\n"
                f"Scarica da: https://github.com/VisDrone/VisDrone-Dataset"
            )

        # Costruisce la lista dei sample validi (almeno min_persons persone)
        self.samples = self._build_index(min_persons)
        if not self.samples:
            raise ValueError(
                f"Nessun frame con ≥{min_persons} persone trovato in {self.root}.\n"
                f"Verifica che le annotazioni siano nella cartella annotations/."
            )

        print(f"VisDroneLoader: {len(self.samples)} frame validi in {self.root.name}")

    def _build_index(self, min_persons: int) -> List[Tuple[Path, Path]]:
        """Indicizza i frame che hanno almeno min_persons persone annotate."""
        samples = []
        img_paths = sorted(self.img_dir.glob("*.jpg")) + \
                    sorted(self.img_dir.glob("*.png"))

        for img_path in img_paths:
            ann_path = self.ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue
            bboxes = self._parse_annotation(ann_path,
                                            orig_w=1, orig_h=1,  # solo per contare
                                            count_only=True)
            if len(bboxes) >= min_persons:
                samples.append((img_path, ann_path))

        return samples

    def _parse_annotation(self, ann_path: Path,
                           orig_w: int, orig_h: int,
                           count_only: bool = False) -> List[Tuple[int,int,int,int]]:
        """
        Legge il file .txt di VisDrone e restituisce bbox in pixel (x1,y1,x2,y2)
        ridimensionati a self.img_size.

        Formato riga:
            bbox_left, bbox_top, bbox_width, bbox_height, score, category, trunc, occl
        """
        bboxes = []
        scale_x = self.img_size / max(orig_w, 1)
        scale_y = self.img_size / max(orig_h, 1)

        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    category    = int(parts[5])
                except ValueError:
                    continue

                if category not in PERSON_CATEGORIES:
                    continue
                if w * h < MIN_BBOX_AREA and not count_only:
                    continue   # skip bbox troppo piccoli

                if count_only:
                    bboxes.append((0, 0, 0, 0))
                    continue

                # Scala e converti in x1,y1,x2,y2
                x1 = int(np.clip(x * scale_x,       0, self.img_size - 1))
                y1 = int(np.clip(y * scale_y,       0, self.img_size - 1))
                x2 = int(np.clip((x + w) * scale_x, 1, self.img_size))
                y2 = int(np.clip((y + h) * scale_y, 1, self.img_size))

                if x2 > x1 and y2 > y1:
                    bboxes.append((x1, y1, x2, y2))

        return bboxes

    def get_sample(self, idx: int) -> Tuple[Image.Image, List[Tuple[int,int,int,int]]]:
        """
        Restituisce (immagine PIL 640×640, lista di bbox persona).

        Ogni bbox è (x1, y1, x2, y2) in pixel dell'immagine ridimensionata.
        """
        img_path, ann_path = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        bboxes  = self._parse_annotation(ann_path, orig_w, orig_h)
        return img_pil, bboxes

    def iter_batches(self, batch_size: int = 4,
                     shuffle: bool = True,
                     max_samples: Optional[int] = None
                     ) -> Iterator[Tuple[List[Image.Image],
                                         List[List[Tuple[int,int,int,int]]]]]:
        """
        Iteratore che yield (imgs, bboxes_per_img) in batch casuali.

        Args:
            batch_size:  immagini per batch
            shuffle:     mescola l'ordine ad ogni epoch
            max_samples: limita il numero totale di sample (utile per test rapidi)
        """
        indices = list(range(len(self.samples)))
        if max_samples is not None:
            indices = indices[:max_samples]
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            imgs, bboxes_list = [], []
            for i in batch_idx:
                img, bboxes = self.get_sample(i)
                if bboxes:   # skip se dopo il filtro non ci sono persone
                    imgs.append(img)
                    bboxes_list.append(bboxes)
            if imgs:
                yield imgs, bboxes_list

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> None:
        """Stampa un riassunto del dataset."""
        total_persons = 0
        for _, ann_path in self.samples[:min(100, len(self.samples))]:
            bboxes = self._parse_annotation(ann_path, 1, 1, count_only=True)
            total_persons += len(bboxes)
        avg = total_persons / min(100, len(self.samples))
        print(f"\nVisDrone dataset summary:")
        print(f"  Frame totali:       {len(self.samples)}")
        print(f"  Persone/frame (avg):{avg:.1f}  (campione di 100 frame)")
        print(f"  img_size:           {self.img_size}×{self.img_size}")
