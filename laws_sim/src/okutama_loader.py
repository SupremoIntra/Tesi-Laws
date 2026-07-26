"""
Okutama-Action DataLoader per LAWS-SIM.

Struttura reale confermata (drone1/2 x mattina/pomeriggio x frame estratti,
label separate per risoluzione nativa):
    okutama_root/
    ├── Drone1/Morning/Extracted-Frames-1280x720/<video>/<frame_num>.jpg
    ├── Drone1/Noon/Extracted-Frames-1280x720/<video>/<frame_num>.jpg
    ├── Drone2/Morning/Extracted-Frames-1280x720/<video>/<frame_num>.jpg
    ├── Drone2/Noon/Extracted-Frames-1280x720/<video>/<frame_num>.jpg
    └── Labels/SingleActionLabels/3840x2160/<video>.txt

    <video> = "<drone>.<time>.<scenario>" (es. "1.1.7"): drone in {1,2},
    time 1=Morning 2=Noon (doc ufficiale), scenario intero libero.
    Frame file: "<frame_num>.jpg", nessun padding (confermato: 0.jpg, 1.jpg, ...).

Formato annotazioni (fonte: okutama-action.org, SingleActionLabels — un'unica
riga per persona per frame, colonne azione già escluse a monte scegliendo
questo file invece di MultiActionLabels):
    TrackID xmin ymin xmax ymax frame lost occluded generated "label" [azioni...]

    - Coordinate in spazio NATIVO 3840x2160 (confermato dal nome cartella
      label + "video resolution (3840x2160)" nella doc ufficiale) — NON
      1280x720. Lo scaling verso il canvas finale usa 3840x2160 come
      riferimento, saltando il passaggio intermedio 1280x720 (che serve
      solo a leggere i pixel del frame, non a definire la scala bbox).
    - label sempre "Person" (verificato comunque, difensivo).
    - lost=1 → fuori schermo, SCARTATA (nessun contenuto visivo).
    - occluded/generated → NON filtrati, stessa scelta di VisDroneLoader
      (parita' di trattamento tra i due dataset).
    - Colonne azione (11+) → ignorate (istruzione testuale della doc
      ufficiale, non inferenza).

RISOLUZIONE (decisione di sessione): img_size=1280 quadrato, resize "a
stretch" dal frame 1280x720 nativo — stessa filosofia di VisDroneLoader
(ignora aspect ratio), per parita' metodologica tra i due dataset, non
per fedelta' pixel ottimale. Fattore di stretch verticale ~1.78x da
tenere a mente nella ricalibrazione soglie 60px/80px (decisione
deferita al preflight, non ancora presa).

NOTA su MIN_BBOX_AREA: il filtro e' applicato in spazio NATIVO (3840x2160),
PRIMA dello scaling — stessa convenzione di VisDroneLoader (che filtra su
pixel nativi dell'immagine originale, prima del resize a 640x640). Il
significato assoluto della soglia dipende quindi dalla risoluzione nativa
di ciascun dataset (non e' una scelta ottimale in senso assoluto), ma e'
la stessa convenzione già in uso — coerenza di metodo, non ottimalita'
per singolo dataset.

Uso (identico a VisDroneLoader):
    loader = OkutamaLoader("data/okutama_train", img_size=1280)
    for img_pil, bboxes in loader.iter_batches(batch_size=4):
        ...
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

# Colonne del formato label Okutama (0-based dopo split su spazi)
_COL_XMIN, _COL_YMIN, _COL_XMAX, _COL_YMAX = 1, 2, 3, 4
_COL_FRAME, _COL_LOST = 5, 6
_COL_LABEL = 9
_MIN_COLUMNS = 10

# Spazio nativo delle coordinate nei file label (confermato: cartella
# "Labels/SingleActionLabels/3840x2160/" + doc ufficiale)
NATIVE_LABEL_WIDTH = 3840
NATIVE_LABEL_HEIGHT = 2160

# Sottocartella dei frame estratti, fissa per questo dataset
_EXTRACTED_FRAMES_DIRNAME = "Extracted-Frames-1280x720"

# 1=Morning, 2=Noon — confermato da doc ufficiale ("1 indica morning, 2 noon")
_TIME_OF_DAY = {"1": "Morning", "2": "Noon"}

MIN_BBOX_AREA = 100  # stessa soglia (e stessa convenzione: nativa, pre-scale) di VisDrone


class OkutamaLoader:
    """
    DataLoader per Okutama-Action (Barekatain et al., 2017), interfaccia
    identica a VisDroneLoader (get_sample, __len__, iter_batches) per
    riuso zero-modifiche di training/eval/bootstrap.

    Costruttore:
        root_dir: cartella con Drone1/, Drone2/, Labels/ (struttura ufficiale
                   dell'archivio scaricato, confermata su disco)
        img_size: lato del canvas quadrato finale (default 1280, decisione
                   di sessione)
        label_subdir: percorso relativo dentro Labels/ (default
                   "SingleActionLabels/3840x2160" — quello effettivamente
                   presente nell'archivio scaricato)
        seed: per riproducibilita' di iter_batches
    """

    def __init__(self, root_dir: str, img_size: int = 1280,
                 label_subdir: str = "SingleActionLabels/3840x2160",
                 seed: int = 42):
        self.root     = Path(root_dir)
        self.img_size = img_size
        self.seed     = seed

        self.label_dir = self.root / "Labels" / label_subdir
        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Cartella label non trovata: {self.label_dir}\n"
                f"Struttura attesa: {self.root}/Labels/{label_subdir}/<video>.txt"
            )

        # samples: (frame_path, [bbox_nativa,...]) — bbox ancora in spazio
        # 3840x2160, scaling fatto in get_sample (lazy, non duplica memoria)
        self.samples: List[Tuple[Path, List[Tuple[int, int, int, int]]]] = []
        self._build_index()

        if not self.samples:
            raise ValueError(
                f"Nessun frame con bbox valida trovato in {self.root}. "
                f"Verifica label_subdir e la presenza dei frame estratti."
            )

        print(f"OkutamaLoader: {len(self.samples)} frame validi in {self.root.name}")

    def _video_image_dir(self, video_name: str) -> Optional[Path]:
        """
        Ricostruisce il path della cartella frame da nome video, es.
        "1.1.7" -> Drone1/Morning/Extracted-Frames-1280x720/1.1.7/
        Ritorna None se il nome non ha il formato atteso (3 interi con
        punti) — scartato senza sollevare eccezione, un video malformato
        non deve fermare l'indicizzazione degli altri.
        """
        parts = video_name.split(".")
        if len(parts) != 3 or parts[0] not in ("1", "2") or parts[1] not in _TIME_OF_DAY:
            return None
        drone_num, time_code = parts[0], parts[1]
        return (self.root / f"Drone{drone_num}" / _TIME_OF_DAY[time_code]
                / _EXTRACTED_FRAMES_DIRNAME / video_name)

    def _build_index(self) -> None:
        """Parsing per-video: un .txt -> frame validi con bbox, merge con i frame reali su disco."""
        for ann_path in sorted(self.label_dir.glob("*.txt")):
            video_name = ann_path.stem
            image_dir = self._video_image_dir(video_name)
            if image_dir is None or not image_dir.exists():
                continue  # video senza cartella frame corrispondente: scartato

            frame_boxes = self._parse_video_annotations(ann_path)

            for frame_num, bboxes in frame_boxes.items():
                frame_path = image_dir / f"{frame_num}.jpg"
                if bboxes and frame_path.exists():
                    self.samples.append((frame_path, bboxes))

    def _parse_video_annotations(
        self, ann_path: Path
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Legge un file SingleActionLabels e raggruppa le bbox per frame.
        Bbox in pixel NATIVI 3840x2160 — nessuno scaling qui.
        """
        frame_boxes: Dict[int, List[Tuple[int, int, int, int]]] = {}

        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < _MIN_COLUMNS:
                    continue

                label = parts[_COL_LABEL].strip('"')
                if label != "Person":
                    continue  # difensivo: dataset a singola classe

                try:
                    lost = int(parts[_COL_LOST])
                except ValueError:
                    continue
                if lost == 1:
                    continue  # fuori schermo

                try:
                    x1 = int(float(parts[_COL_XMIN]))
                    y1 = int(float(parts[_COL_YMIN]))
                    x2 = int(float(parts[_COL_XMAX]))
                    y2 = int(float(parts[_COL_YMAX]))
                    frame_num = int(parts[_COL_FRAME])
                except ValueError:
                    continue

                if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                    continue  # rumore, stessa soglia/convenzione di VisDrone

                if x2 > x1 and y2 > y1:
                    frame_boxes.setdefault(frame_num, []).append((x1, y1, x2, y2))

        return frame_boxes

    def get_sample(self, idx: int) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        """
        Restituisce (immagine PIL img_size×img_size, lista bbox persona).

        Scaling in un solo passaggio: da spazio nativo label (3840x2160)
        al canvas finale img_size — non passa per 1280x720 intermedio,
        che serve solo a caricare i pixel del frame.
        """
        frame_path, bboxes_native = self.samples[idx]
        img_pil = Image.open(frame_path).convert("RGB")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)

        scale_x = self.img_size / NATIVE_LABEL_WIDTH
        scale_y = self.img_size / NATIVE_LABEL_HEIGHT

        bboxes_scaled = []
        for x1, y1, x2, y2 in bboxes_native:
            sx1 = int(np.clip(x1 * scale_x, 0, self.img_size - 1))
            sy1 = int(np.clip(y1 * scale_y, 0, self.img_size - 1))
            sx2 = int(np.clip(x2 * scale_x, 1, self.img_size))
            sy2 = int(np.clip(y2 * scale_y, 1, self.img_size))
            if sx2 > sx1 and sy2 > sy1:
                bboxes_scaled.append((sx1, sy1, sx2, sy2))

        return img_pil, bboxes_scaled

    def iter_batches(self, batch_size: int = 4, shuffle: bool = True,
                      max_samples: Optional[int] = None
                      ) -> Iterator[Tuple[List[Image.Image],
                                          List[List[Tuple[int, int, int, int]]]]]:
        """Identico a VisDroneLoader.iter_batches — stessa firma, stesso comportamento."""
        import random
        random.seed(self.seed)

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
                if bboxes:
                    imgs.append(img)
                    bboxes_list.append(bboxes)
            if imgs:
                yield imgs, bboxes_list

    def __len__(self) -> int:
        return len(self.samples)