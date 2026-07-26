"""
Okutama-Action DataLoader per LAWS-SIM.

Stessa interfaccia di VisDroneLoader (get_sample, __len__, iter_batches),
cosi' il resto della pipeline (training, eval, bootstrap CI) non richiede
NESSUNA modifica — cambia solo il dataset (richiesta esplicita del
relatore: confronto side-by-side, non accorpamento).

Formato annotazioni Okutama (fonte: okutama-action.org, sezione Dataset
Download — non VisDrone-style, quindi NON riusa il parser esistente):
    TrackID xmin ymin xmax ymax frame lost occluded generated "label" [azioni...]

    - label e' sempre "Person" (singola classe, come la nostra PERSON_CATEGORIES
      di VisDrone, ma qui non serve nemmeno un filtro per categoria).
    - lost=1 → bbox fuori dallo schermo (nessun contenuto visivo reale):
      SCARTATA. Analogo concettuale del filtro MIN_BBOX_AREA di VisDrone
      (una bbox "lost" non e' un bersaglio piu' piccolo, e' un bersaglio
      assente — includerla vorrebbe dire disegnare la patch sul nulla).
    - occluded, generated (interpolato): NON filtrati, stessa scelta di
      VisDrone (che non filtra per truncation/occlusion, solo per area
      minima) — parita' di trattamento tra i due dataset.
    - Colonne azione (11+): ignorate. Confermato dalla documentazione
      ufficiale ("For pedestrian detection task, the columns describing
      the actions should be ignored") — non e' una nostra semplificazione,
      e' l'uso previsto del dataset per detection puro.

RISOLUZIONE (decisione presa in sessione, NON assunta):
    img_size=1280 (quadrato), stessa filosofia di resize "a stretch" di
    VisDroneLoader (ignora aspect ratio nativo). Scelta deliberata per
    PARITA' METODOLOGICA tra i due dataset (stesso tipo di preprocessing,
    comparabilita' del metodo), non perche' sia la resa pixel ottimale
    per Okutama — i frame nativi pre-estratti sono 1280x720 (16:9), quindi
    lo stretch a 1280x1280 introduce un fattore di allungamento verticale
    (~1.78x) che va tenuto a mente quando si ricalibrano le soglie
    60px/80px (decisione lasciata al post-preflight, come stabilito).

STRUTTURA CARTELLE ASSUNTA (da verificare dopo unzip reale degli archivi
ufficiali — non ho visibilita' diretta sul contenuto degli zip):
    okutama_root/
    ├── images/
    │   └── <video_name>/          es. "1.1.7"/
    │       └── <frame_pattern>    es. frame_000001.jpg (pattern configurabile)
    └── annotations/
        └── <video_name>.txt       una riga per bbox per frame, tutto il video

Se la struttura reale differisce (es. frame in cartella flat con nome
"<video>_<frame>.jpg"), va aggiornato solo _build_index/_frame_image_path
— il resto della classe (parsing, scaling, interfaccia pubblica) resta
valido.

Uso base (identico a VisDroneLoader):
    loader = OkutamaLoader("data/okutama_train", img_size=1280)
    for img_pil, bboxes in loader.iter_batches(batch_size=4):
        ...
"""

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

# Colonne del formato label Okutama (indice 0-based dopo lo split):
# 0=TrackID 1=xmin 2=ymin 3=xmax 4=ymax 5=frame 6=lost 7=occluded 8=generated 9=label
_COL_XMIN, _COL_YMIN, _COL_XMAX, _COL_YMAX = 1, 2, 3, 4
_COL_FRAME, _COL_LOST = 5, 6
_COL_LABEL = 9
_MIN_COLUMNS = 10

MIN_BBOX_AREA = 100  # stessa soglia di VisDrone (10x10 px), parita' di trattamento


class OkutamaLoader:
    """
    DataLoader per Okutama-Action (Barekatain et al., 2017), interfaccia
    identica a VisDroneLoader per riuso zero-modifiche del resto della
    pipeline (patch_optimizer, simulator, cli).

    Costruttore:
        root_dir:      cartella con images/ e annotations/ (vedi docstring
                        di modulo per la struttura assunta)
        img_size:      lato del canvas quadrato dopo resize (default 1280,
                        decisione di sessione — vedi nota metodologica sopra)
        frame_glob:    pattern glob per i file immagine dentro ogni
                        sottocartella video (default "*.jpg")
        seed:          per riproducibilita' (usato da iter_batches)

    Nota sul design (parita' con VisDroneLoader):
        Il parsing e' per-video (un file label = tutte le righe di tutti i
        frame di quel video), quindi l'indice interno raggruppa le bbox per
        frame UNA SOLA VOLTA in __init__ (costo O(righe totali), non
        O(frame x righe) come sarebbe ri-parsando il file ad ogni
        get_sample) — stesso principio di efficienza di plot_k_selection.py
        (cumulative sum invece di ricalcolo per ogni K).
    """

    def __init__(self, root_dir: str, img_size: int = 1280,
                 frame_glob: str = "*.jpg", seed: int = 42):
        self.root       = Path(root_dir)
        self.img_size   = img_size
        self.frame_glob = frame_glob
        self.seed       = seed

        self.img_dir = self.root / "images"
        self.ann_dir  = self.root / "annotations"

        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Struttura non trovata in {self.root}.\n"
                f"Attese: {self.img_dir} e {self.ann_dir}\n"
                f"Verifica la struttura reale dopo l'unzip degli archivi "
                f"ufficiali (okutama-action.org) — questa classe assume "
                f"images/<video>/<frame>.jpg e annotations/<video>.txt."
            )

        # samples: lista di (video_dir, frame_path, [bbox,...]) — una entry
        # per frame che ha almeno 1 bbox valida (bbox già in pixel nativi,
        # scaling a img_size fatto lazy in get_sample per non duplicare
        # memoria su migliaia di frame).
        self.samples: List[Tuple[Path, List[Tuple[int, int, int, int]]]] = []
        self._build_index()

        if not self.samples:
            raise ValueError(
                f"Nessun frame con bbox valida trovato in {self.root}. "
                f"Verifica che i file .txt in annotations/ seguano il "
                f"formato ufficiale Okutama (10+ colonne, vedi docstring)."
            )

        print(f"OkutamaLoader: {len(self.samples)} frame validi in {self.root.name}")

    def _build_index(self) -> None:
        """
        Parsing per-video: un file .txt -> dict {frame_num: [bbox,...]},
        poi merge con i file immagine realmente presenti su disco (un
        video puo' avere piu' frame annotati di quanti estratti, o
        viceversa — usiamo l'intersezione, mai un'assunzione silenziosa).
        """
        video_dirs = sorted(d for d in self.img_dir.iterdir() if d.is_dir())

        for video_dir in video_dirs:
            ann_path = self.ann_dir / f"{video_dir.name}.txt"
            if not ann_path.exists():
                continue  # video senza label: scartato, non e' un errore fatale

            frame_boxes = self._parse_video_annotations(ann_path)

            for frame_path in sorted(video_dir.glob(self.frame_glob)):
                frame_num = self._frame_number_from_filename(frame_path.name)
                if frame_num is None or frame_num not in frame_boxes:
                    continue
                bboxes = frame_boxes[frame_num]
                if bboxes:  # skip frame senza bbox valide (tutte "lost" o area<soglia)
                    self.samples.append((frame_path, bboxes))

    @staticmethod
    def _frame_number_from_filename(filename: str) -> Optional[int]:
        """
        Estrae il numero di frame dal nome file (es. 'frame_000123.jpg' -> 123).
        Pattern da CONFERMARE sulla struttura reale post-unzip — questo
        regex cattura la prima sequenza di cifre nel nome, funziona per le
        convenzioni piu' comuni ma va verificato sul primo video reale.
        """
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else None

    def _parse_video_annotations(
        self, ann_path: Path
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Legge un file label Okutama e raggruppa le bbox per numero di frame.
        Bbox in pixel NATIVI (nessuno scaling qui — fatto in get_sample,
        dove si conosce la dimensione reale dell'immagine caricata).
        """
        frame_boxes: Dict[int, List[Tuple[int, int, int, int]]] = {}

        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < _MIN_COLUMNS:
                    continue

                label = parts[_COL_LABEL].strip('"')
                if label != "Person":
                    continue  # difensivo: il dataset ha una sola classe, ma verifichiamo

                try:
                    lost = int(parts[_COL_LOST])
                except ValueError:
                    continue
                if lost == 1:
                    continue  # fuori schermo: nessun contenuto visivo reale

                try:
                    x1 = int(float(parts[_COL_XMIN]))
                    y1 = int(float(parts[_COL_YMIN]))
                    x2 = int(float(parts[_COL_XMAX]))
                    y2 = int(float(parts[_COL_YMAX]))
                    frame_num = int(parts[_COL_FRAME])
                except ValueError:
                    continue

                if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                    continue  # stessa soglia rumore di VisDrone

                if x2 > x1 and y2 > y1:
                    frame_boxes.setdefault(frame_num, []).append((x1, y1, x2, y2))

        return frame_boxes

    def get_sample(self, idx: int) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        """
        Restituisce (immagine PIL img_size×img_size, lista bbox persona).

        Resize "a stretch" (ignora aspect ratio nativo 16:9), STESSA
        convenzione di VisDroneLoader.get_sample — parita' metodologica
        tra i due dataset, vedi nota nella docstring di modulo.
        """
        frame_path, bboxes_native = self.samples[idx]
        img_pil = Image.open(frame_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)

        scale_x = self.img_size / max(orig_w, 1)
        scale_y = self.img_size / max(orig_h, 1)

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