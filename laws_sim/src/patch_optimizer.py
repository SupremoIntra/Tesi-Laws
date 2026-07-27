"""
Adversarial Patch Optimizer — Untargeted Asymptotic Attack, Full EoT & MPS Optimized

Architettura:
    - Gradient Accumulation per stabilità su hardware limitato
    - Loss asintotica: -log(1 - mean(confidence) + epsilon)
    - EoT completo (16 trasformazioni per immagine singola)
    - Checkpoint resumabile e recovery da interruzione

Riferimenti Accademici:
    [1] Carlini & Wagner (2017). "Towards Evaluating the Robustness of Neural Networks".
    [2] Wu et al. (2020). "Making an Invisibility Cloak: Real World Adversarial Attacks 
        on Object Detectors". 
    [3] Athalye et al. (2017). "Synthesizing Robust Adversarial Examples".
    [4] Thys et al. (2019). "Fooling automated surveillance cameras: adversarial patches 
        to attack person detection".
    [5] Brown et al. (2017). "Adversarial Patch".

Autore: LAWS-SIM Framework
Data: 2026
"""

import os

# grid_sampler_2d_backward non è implementato su MPS (usato dall'EoT per
# rotazione/scala/aspect ratio). Questo fa girare solo quell'operatore su
# CPU, il resto (YOLO) resta su MPS. Deve stare prima di "import torch".
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    PATCH_H, PATCH_W, PATCH_LR, PATCH_STEPS,
    EOT_N_TRANSFORMS, PERSON_CLASS_ID, IMG_SIZE,
    BATCH_SIZE_PHYSICAL, GRADIENT_ACCUMULATION_STEPS,
    TV_WEIGHT, CHECKPOINT_EVERY, EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_WINDOW, CHECKPOINT_FILE, BEST_PATCH_FILE,
    METRICS_JSON_FILE, LOSS_TOP_K
)

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[WARNING] ultralytics non installato. Funzionalità limitata.")


def get_chest_bbox_proportional(
    person_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    """
    Calcola bounding box proporzionale per posizionamento patch (Tactical Vest).
    
    Estende la bbox della persona al 50% della larghezza e 40% dell'altezza,
    simulando un giubbotto tattico che massimizza la superficie di attacco
    mantenendo realismo fisico.
    
    Args:
        person_bbox: Bounding box originale (x1, y1, x2, y2)
        img_w: Larghezza immagine
        img_h: Altezza immagine
        
    Returns:
        Tuple (px1, py1, px2, py2) della bbox estesa, clampata ai bordi immagine
        
    Note:
        Il posizionamento è centrato sul petto (33% dall'alto) per massimizzare
        la visibilità da prospettive drone tipiche (VisDrone dataset).
    """
    x1, y1, x2, y2 = person_bbox
    w, h = x2 - x1, y2 - y1
    
    # Estensione tattica: 50% larghezza, 40% altezza
    pw = max(int(w * 0.50), 4)
    ph = max(int(h * 0.40), 4)
    
    # Centro sul petto (non centro geometrico)
    cx = x1 + w // 2
    cy = y1 + int(h * 0.33)
    
    # Calcolo bbox estesa con clamping
    px1 = max(0, cx - pw // 2)
    py1 = max(0, cy - ph // 2)
    px2 = min(img_w, px1 + pw)
    py2 = min(img_h, py1 + ph)
    
    return (px1, py1, px2, py2)


def tactical_preflight_check(
    loader,
    low: float = 60.0,
    high: float = 80.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    # TACTICAL FILTER 2026
    Scandisce le annotazioni del trainset (solo header immagine + file di
    testo, NESSUN caricamento/decode dei pixel) e conta:
      1. Annotazioni persona totali (>= MIN_BBOX_AREA, il filtro minimo
         già esistente nel loader)
      2. Annotazioni >= low px (60, il filtro anti-downsampling già in uso)
      3. Annotazioni >= high px (80, soglia "tatticamente rilevante")

    Da lanciare PRIMA del training, per avere un dato empirico su quanto
    del dataset è coerente con lo scenario del simulatore (ingaggio
    ravvicinato, DRONE_ALTITUDE_M=10) prima di allenare la patch.

    Returns:
        dict con conteggi e percentuali, stampato a schermo se verbose.
    """
    from PIL import Image as _Image

    total = 0
    above_low = 0
    above_high = 0

    for img_path, ann_path in loader.samples:
        with _Image.open(img_path) as im:
            orig_w, orig_h = im.size
        bboxes = loader._parse_annotation(ann_path, orig_w, orig_h)
        for b in bboxes:
            h = b[3] - b[1]
            total += 1
            if h >= low:
                above_low += 1
            if h >= high:
                above_high += 1

    pct_low = (above_low / total * 100) if total else 0.0
    pct_high = (above_high / total * 100) if total else 0.0

    result = {
        "n_annotazioni_totali": total,
        "n_annotazioni_utilizzabili_60px": above_low,
        "n_annotazioni_tattiche_80px": above_high,
        "pct_utilizzabili_60px": pct_low,
        "pct_tattiche_80px": pct_high,
    }

    if verbose:
        print("\n[TACTICAL FILTER 2026] Pre-flight check dataset:")
        print(f"  Annotazioni persona totali:              {total}")
        print(f"  Annotazioni >= {low:.0f}px (utilizzabili):       {above_low}  ({pct_low:.1f}%)")
        print(f"  Annotazioni >= {high:.0f}px (tatticamente rilevanti): {above_high}  ({pct_high:.1f}%)")
        print(f"  --> Il {100-pct_high:.1f}% delle annotazioni e' sotto la soglia tattica.\n")

    return result


class PatchOptimizer:
    """
    Ottimizzatore per Universal Adversarial Patch con EoT completo.
    
    Implementa un attacco untargeted asintotico che minimizza la confidenza
    media della classe target attraverso una loss logaritmica, garantendo
    convergenza stabile anche con gradienti di piccola entità.
    
    Architettura:
        - Parametrizzazione sigmoide per evitare dead pixels ai bordi
        - O(1) Canvas generation per efficienza computazionale
        - Gradient Accumulation per stabilità su hardware limitato
        - Early Stopping basato su media mobile della loss
        
    Attributes:
        patch_logits: Tensor dei logits della patch (C, H, W)
        TV_WEIGHT: Peso della Total Variation Loss per regolarizzazione spaziale
    """
    
    def __init__(
        self,
        patch_h: int = PATCH_H,
        patch_w: int = PATCH_W,
        model_path: str = "yolov8n.pt"
    ):
        """
        Inizializza l'ottimizzatore con parametrizzazione sigmoide.
        
        Args:
            patch_h: Altezza della patch in pixel
            patch_w: Larghezza della patch in pixel
            model_path: Path al modello YOLO pre-addestrato
            
        Raises:
            RuntimeError: Se ultralytics non è installato
            
        Note:
            La parametrizzazione sigmoide (logits → sigmoid → [0,1]) è critica
            per evitare gradient vanishing ai bordi della patch e garantire
            che tutti i pixel contribuiscano all'ottimizzazione.
        """
        if not HAS_YOLO:
            raise RuntimeError("pip install ultralytics")
            
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.model_path = model_path
        self._yolo = None
        
        # Parametrizzazione Sigmoide: evita dead pixels e gradient vanishing
        # Inizializzazione piccola (std=0.1) per evitare saturazione iniziale
        device = self._get_device()
        self.patch_logits = (
            torch.randn(3, patch_h, patch_w, device=device) * 0.1
        ).requires_grad_(True)
        
        # Metriche per logging
        self.training_metrics: Dict[str, List[Any]] = {
            "timestamp": [],
            "step": [],
            "loss": [],
            "tv_loss": [],
            "confidence": [],
            "learning_rate": [],
            "grad_norm": [],
            "main_loss_grad_norm": []
        }
    
    def _get_device(self) -> str:
        """
        Determina il device ottimale per l'hardware disponibile.
        
        Returns:
            "mps" per Apple Silicon, "cuda" per NVIDIA, "cpu" come fallback
            
        Note:
            Preferiamo MPS su M4 per speedup ~12x rispetto a CPU, mantenendo
            Float32 per stabilità dei gradienti nelle operazioni spaziali.
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_model(self):
        """
        Lazy initialization del modello YOLO con freezing dei pesi.
        
        Returns:
            Istanza YOLO pronta per inferenza (eval mode, no grad)
            
        Note:
            I pesi di YOLO sono congelati (requires_grad=False) poiché
            stiamo ottimizzando solo la patch, non il detector.
        """
        if self._yolo is None:
            device = self._get_device()
            self._yolo = _YOLO(self.model_path)
            self._yolo.to(device)
            
            # Congela tutti i pesi del modello
            for p in self._yolo.model.parameters():
                p.requires_grad_(False)
            self._yolo.model.eval()
            
            # Warmup con dummy forward pass
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
            _ = self._yolo.model(dummy)
            
        return self._yolo
    
    @staticmethod
    def _build_spatial_mask(
        person_bbox: Tuple[int, int, int, int],
        img_size: int = 640,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Costruisce maschera spaziale per gli stride grid di YOLOv8.
        
        YOLOv8 usa tre scale di predizione (stride 8, 16, 32). Questa funzione
        crea una maschera booleana che identifica quali anchor grid cells
        cadono all'interno della bounding box della persona.
        
        Args:
            person_bbox: Bounding box della persona (x1, y1, x2, y2)
            img_size: Dimensione immagine (default 640)
            device: Device per il tensor
            
        Returns:
            Tensor booleano concatenato per tutti gli stride
            
        Note:
            La maschera è essenziale per calcolare la loss solo sulle
            predizioni rilevanti, riducendo il rumore del gradiente.
        """
        x_min, y_min, x_max, y_max = person_bbox
        strides = [8, 16, 32]
        mask_list = []
        
        for s in strides:
            grid_size = img_size // s
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_size, device=device),
                torch.arange(grid_size, device=device),
                indexing='ij'
            )
            
            # Coordinate centri delle celle grid
            cx = (grid_x + 0.5) * s
            cy = (grid_y + 0.5) * s
            
            # Mask: True se centro cella è dentro bbox
            in_box = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
            mask_list.append(in_box.flatten())
            
        return torch.cat(mask_list, dim=0)
    
    @staticmethod
    def _asymptotic_loss(
        torch_model: torch.nn.Module,
        batch: torch.Tensor,
        spatial_mask: torch.Tensor,
        person_class: int = PERSON_CLASS_ID,
        epsilon: float = 1e-6,
        top_k: int = LOSS_TOP_K,
        cell_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Loss asintotica untargeted: -log(1 - top_k_mean(confidence) + epsilon).
        
        Questa loss spinge la confidenza asintoticamente verso zero, penalizzando
        pesantemente i casi in cui YOLO mantiene una confidenza residua elevata.
        
        Aggregazione: media sulle top_k celle più confidenti dentro la spatial
        mask, non su tutte. Thys et al. (2019) hanno mostrato sperimentalmente
        che minimizzare il MASSIMO objectness score dell'immagine batte sia
        l'attacco alla sola classe sia la combinazione di entrambi. Un massimo
        puro, nel nostro caso, attaccherebbe solo una persona nei frame
        multi-target (fino a 3, MAX_TARGETS_PER_FRAME); top_k generalizza
        l'idea concentrando il gradiente sulle celle realmente vicine a una
        detection reale, invece di diluirlo sulle centinaia di celle di
        sfondo che la spatial mask include per costruzione (tre stride,
        intera area del bbox) e che cambiano casualmente da immagine a
        immagine — la causa di rumore diagnosticata misurando grad_norm.
        
        # TACTICAL FILTER 2026
        cell_weights (opzionale): peso [0,1] per cella, stessa forma di
        spatial_mask, derivato dall'altezza del bbox del target che ha
        generato quella cella (vedi _tactical_weight). Se fornito, la media
        top-K diventa una media PESATA: le celle vengono comunque scelte
        per confidenza grezza (dove YOLO guarda davvero), ma il loro
        contributo alla loss finale è scalato dalla rilevanza tattica.
        Se None, comportamento identico alla versione precedente (retro-
        compatibile: nessuna modifica per chi non passa questo argomento).
        
        Args:
            torch_model: Modello YOLO (PyTorch nn.Module)
            batch: Batch di immagini con patch applicata (N, C, H, W)
            spatial_mask: Maschera booleana per selezionare predizioni rilevanti
            person_class: ID della classe target (default: person)
            epsilon: Valore piccolo per evitare log(0)
            top_k: Numero di celle più confidenti su cui mediare
            cell_weights: peso tattico per cella (TACTICAL FILTER 2026)
            
        Returns:
            Scalar loss tensor
        """
        raw = torch_model(batch)
        preds = raw[0] if isinstance(raw, (tuple, list)) else raw
        
        # Estrai confidenze classe target e applica sigmoid
        person_scores = torch.sigmoid(preds[:, 4 + person_class, :])
        
        # Seleziona solo le predizioni dentro la bbox (spatial mask)
        masked_scores = person_scores[:, spatial_mask]
        
        # Top-K invece di mean su tutte le celle: concentra il gradiente
        # sulle celle vicine a una detection reale (vedi Thys et al. 2019)
        k = min(top_k, masked_scores.shape[1])
        top_vals, top_idx = masked_scores.topk(k, dim=1)

        if cell_weights is not None:
            # TACTICAL FILTER 2026: media pesata sulle celle selezionate.
            # Le celle restano scelte per confidenza grezza (top_idx sopra),
            # ma un target < 150px (peso < 1) conta proporzionalmente meno
            # nella media finale, senza alterare la probabilità stessa
            # (niente conf*peso, che romperebbe la semantica di -log(1-x)).
            masked_weights = cell_weights[spatial_mask]
            masked_weights_exp = masked_weights.unsqueeze(0).expand(masked_scores.shape[0], -1)
            top_weights = torch.gather(masked_weights_exp, 1, top_idx)
            weight_total = top_weights.sum().clamp_min(epsilon)
            mean_conf = (top_vals * top_weights).sum() / weight_total
        else:
            mean_conf = top_vals.mean()
        
        # Loss asintotica: -log(1 - mean_conf + epsilon)
        loss = -torch.log(1.0 - mean_conf + epsilon)
        
        return loss
    
    @staticmethod
    def _tactical_weight(bbox_height: float, low: float = 60.0, high: float = 150.0) -> float:
        """
        # TACTICAL FILTER 2026
        Peso [0,1] di rilevanza tattica in funzione dell'altezza del bbox,
        coerente con lo scenario del simulatore (DRONE_ALTITUDE_M=10,
        ingaggio ravvicinato — non sorveglianza ad area larga).

        - height < 60px: peso 0.0 — target fuori dal filtro anti-downsampling
          già esistente, comunque non dovrebbe arrivare qui.
        - 60px <= height < 150px: rampa lineare 0.0 -> 1.0.
        - height >= 150px: peso 1.0 — target pienamente tattico.

        Nota: NON scarta l'immagine né il target dal canvas (la patch viene
        comunque posizionata, per realismo fisico) — esclude solo il
        contributo di quel target al calcolo della loss se il peso è 0,
        e lo attenua nella zona di rampa.
        """
        if bbox_height < low:
            return 0.0
        if bbox_height >= high:
            return 1.0
        return (bbox_height - low) / (high - low)
    
    @staticmethod
    def _visdrone_eot(
        patch: torch.Tensor,
        n_eot: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expectation over Transformation (EoT) per robustezza fisica.
        
        Applica n_eot trasformazioni stocastiche alla patch per simulare
        variazioni reali in deployment: colore, luminosità, rumore, rotazione,
        scala e aspect ratio (simulazione pitch drone).
        
        Args:
            patch: Tensor patch (C, H, W) in [0, 1]
            n_eot: Numero di trasformazioni da applicare
            
        Returns:
            Tuple (patch_rgb, patch_mask) entrambi (n_eot, C, H, W)
            
        Note:
            Le trasformazioni seguono Athalye et al. (2017) e includono:
            - Variazione colore (0.6-1.4x) e luminosità (±0.2)
            - Rumore gaussiano (σ=0.05)
            - Rotazione (±15°)
            - Scala (0.4-0.9x) con aspect ratio variabile (0.7-1.3)
            - Traslazione (±10%)
            
            Questo garantisce che la patch funzioni in condizioni reali,
            non solo in ambiente digitale perfetto.
        """
        device = patch.device
        _, h, w = patch.shape
        
        # Espandi patch per n_eot trasformazioni
        patch_batch = patch.unsqueeze(0).expand(n_eot, -1, -1, -1)
        
        # Variazione colore e luminosità
        c_factor = torch.empty(n_eot, 1, 1, 1, device=device).uniform_(0.6, 1.4)
        b_factor = torch.empty(n_eot, 1, 1, 1, device=device).uniform_(-0.2, 0.2)
        patch_rgb = patch_batch * c_factor + b_factor
        
        # Rumore gaussiano per simulare compressione JPEG e sensori imperfetti
        noise = torch.randn_like(patch_rgb) * 0.05
        patch_rgb = torch.clamp(patch_rgb + noise, 0.0, 1.0)
        
        # Alpha channel per masking
        alpha = torch.ones(n_eot, 1, h, w, device=device)
        patch_rgba = torch.cat([patch_rgb, alpha], dim=1)
        
        # Parametri trasformazione affine
        angles = torch.empty(n_eot, device=device).uniform_(-15.0, 15.0) * (np.pi / 180.0)
        scale_x = torch.empty(n_eot, device=device).uniform_(0.4, 0.9)
        aspect_ratio = torch.empty(n_eot, device=device).uniform_(0.7, 1.3)
        scale_y = scale_x * aspect_ratio
        tx = torch.empty(n_eot, device=device).uniform_(-0.1, 0.1)
        ty = torch.empty(n_eot, device=device).uniform_(-0.1, 0.1)
        
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        # Matrice affine con scale indipendenti X/Y per simulare pitch drone
        theta = torch.zeros(n_eot, 2, 3, device=device)
        theta[:, 0, 0] = cos_a / scale_x
        theta[:, 0, 1] = -sin_a / scale_x
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a / scale_y
        theta[:, 1, 1] = cos_a / scale_y
        theta[:, 1, 2] = ty
        
        # Applica trasformazione affine
        grid = F.affine_grid(theta, patch_rgba.size(), align_corners=False)
        transformed = F.grid_sample(
            patch_rgba, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        trans_rgb = transformed[:, :3, :, :]
        trans_mask = transformed[:, 3:, :, :]
        
        return trans_rgb, trans_mask
    
    @staticmethod
    def _tv_loss(patch: torch.Tensor) -> torch.Tensor:
        """
        Total Variation Loss per regolarizzazione spaziale.
        
        Penalizza variazioni ad alta frequenza tra pixel adiacenti, favorendo
        pattern smooth fisicamente stampabili (Thys et al. 2019).
        
        Args:
            patch: Tensor patch (C, H, W)
            
        Returns:
            Scalar TV loss
            
        Note:
            TV_WEIGHT=0.1 genera pattern low-frequency che sopravvivono
            all'interpolazione bilineare di F.grid_sample durante EoT.
            Valori troppo bassi (0.0001) producono rumore high-frequency
            distrutto dal filtering implicito delle trasformazioni.
        """
        dx = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
        dy = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
        return dx + dy
    
    @staticmethod
    def _get_person_conf(results) -> float:
        """
        Estrae la massima confidenza per la classe person dai risultati YOLO.
        
        Args:
            results: Output di YOLO inference
            
        Returns:
            Massima confidenza per classe person, 0.0 se non rilevato
        """
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                mask = (r.boxes.cls == PERSON_CLASS_ID)
                if mask.any():
                    return float(r.boxes.conf[mask].max())
        return 0.0
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """
        Salva checkpoint dei logits della patch.
        
        Args:
            step: Step corrente del training
            is_best: Se True, salva anche come best patch
        """
        checkpoint = {
            "step": step,
            "patch_logits": self.patch_logits.detach().cpu(),
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(checkpoint, CHECKPOINT_FILE)
        
        if is_best:
            best_patch = torch.sigmoid(self.patch_logits).detach().cpu()
            torch.save(best_patch, BEST_PATCH_FILE)
            print(f"  [Checkpoint] Salvata best patch allo step {step}")
    
    def _load_checkpoint(self) -> int:
        """
        Carica checkpoint se esiste e ripristina i logits.
        
        Returns:
            Step da cui ripartire (0 se nessun checkpoint)
        """
        if os.path.exists(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu")
            device = self._get_device()
            
            self.patch_logits.data = checkpoint["patch_logits"].to(device)
            self.patch_logits.requires_grad_(True)
            
            old_accum = checkpoint.get("gradient_accumulation_steps")
            if old_accum is not None and old_accum != GRADIENT_ACCUMULATION_STEPS:
                print(
                    f"[Checkpoint] ATTENZIONE: questo checkpoint è stato salvato con "
                    f"GRADIENT_ACCUMULATION_STEPS={old_accum}, ora è {GRADIENT_ACCUMULATION_STEPS}. "
                    f"Se vuoi un run pulito con i nuovi iperparametri, interrompi e lancia "
                    f"con --fresh per ripartire da zero."
                )
            
            print(f"[Checkpoint] Ripristinato training dallo step {checkpoint['step']}")
            return checkpoint["step"]
        return 0
    
    def _save_metrics(self):
        """Salva metriche di training in formato JSON per analisi."""
        with open(METRICS_JSON_FILE, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
    
    def _infinite_batches(self, loader, batch_size: int):
        """Generatore infinito di batch dal loader."""
        while True:
            yield from loader.iter_batches(batch_size=batch_size, shuffle=True)
    
    def optimize_universal(
        self,
        loader,
        n_steps: int = PATCH_STEPS,
        lr: float = PATCH_LR,
        n_eot: int = EOT_N_TRANSFORMS,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Training della Universal Adversarial Patch con EoT completo.
        
        Implementa gradient accumulation per stabilità su hardware limitato,
        early stopping basato su media mobile della loss, e checkpoint automatici.
        
        Args:
            loader: Data loader per VisDrone dataset
            n_steps: Numero totale di step di training
            lr: Learning rate iniziale
            n_eot: Numero di trasformazioni EoT per immagine
            verbose: Se True, stampa log dettagliati
            
        Returns:
            Dictionary con patch ottimizzata e storia training:
                - "patch": Tensor patch finale (C, H, W)
                - "loss_history": Lista loss per step
                - "conf_history": Lista (step, confidence) per telemetria
                - "best_step": Step con loss minima
                
        Note:
            Il training usa:
            - Batch fisico = 1, accumulazione su 4 step → batch effettivo = 4
            - Loss asintotica per evitare vanishing gradient
            - Early stopping con pazienza 200 step
            - Checkpoint ogni 100 step e recovery da KeyboardInterrupt
        """
        MAX_TARGETS_PER_FRAME = 3
        
        # Carica checkpoint se esiste
        start_step = self._load_checkpoint()
        
        model = self._get_model()
        torch_model = model.model
        device = self._get_device()
        
        # Optimizer e scheduler
        # T_max è in unità di update reali (chiamate a scheduler.step(),
        # una per update, non una per step raw) — deve essere n_steps
        # diviso per l'accumulo, altrimenti la curva si allunga di un
        # fattore GRADIENT_ACCUMULATION_STEPS e il LR non scende mai
        # a eta_min entro la durata effettiva del run.
        expected_updates = max(1, int(n_steps) // GRADIENT_ACCUMULATION_STEPS)
        optimizer = Adam([self.patch_logits], lr=lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=expected_updates,
            eta_min=lr * 0.1
        )
        
        # Ripristina stato scheduler se checkpoint (in unità di update, non step raw)
        if start_step > 0:
            scheduler.step(start_step // GRADIENT_ACCUMULATION_STEPS)
        
        batch_gen = self._infinite_batches(loader, BATCH_SIZE_PHYSICAL)
        
        loss_history = []
        conf_history = []
        best_loss = float('inf')
        best_step = start_step
        steps_without_improvement = 0
        
        # Buffer per gradient accumulation
        accumulated_loss = 0.0
        accumulation_counter = 0
        
        if verbose:
            print(f"\n[Training] Universal Adversarial Patch | M4 MPS Optimized")
            print(f"Steps: {n_steps} | EoT: {n_eot} | Gradient Accum: {GRADIENT_ACCUMULATION_STEPS}")
            print(f"Batch Effettivo: {BATCH_SIZE_PHYSICAL * GRADIENT_ACCUMULATION_STEPS}")
            print("─" * 70)
        
        try:
            for step in range(start_step, int(n_steps)):
                imgs_pil, bboxes_list = next(batch_gen)
                
                if not any(bboxes_list):
                    continue
                
                # Zero gradienti solo all'inizio del ciclo di accumulazione
                if accumulation_counter == 0:
                    optimizer.zero_grad()
                
                img_pil = imgs_pil[0]
                bboxes = bboxes_list[0]
                
                if not bboxes:
                    continue
                
                # .contiguous() necessario: senza, lo stride creato da
                # .permute() si propaga attraverso .expand() più sotto e
                # rompe il primo conv2d di YOLO su backend MPS.
                img_t = torch.from_numpy(
                    np.array(img_pil).astype(np.float32) / 255.0
                ).permute(2, 0, 1).contiguous().to(device)
                img_size = img_t.shape[-1]  # dinamico: 640 su VisDrone, 1280 su Okutama — nessuna regressione su VisDrone
                # Area-based Sampling: prendi solo i target più grandi
                if len(bboxes) > MAX_TARGETS_PER_FRAME:
                    bboxes = sorted(
                        bboxes,
                        key=lambda b: (b[2]-b[0])*(b[3]-b[1]),
                        reverse=True
                    )[:MAX_TARGETS_PER_FRAME]
                
                # Canvas per composizione patch
                global_canvas_rgb = torch.zeros(n_eot, 3, img_size, img_size, device=device)
                global_canvas_mask = torch.zeros(n_eot, 1, img_size, img_size, device=device)
                global_spatial_mask_list = []
                global_weight_list = []  # TACTICAL FILTER 2026
                targets_in_image = 0
                
                # Genera patch corrente (nuovo grafo computazionale)
                current_patch = torch.sigmoid(self.patch_logits)
                trans_rgb, trans_mask = self._visdrone_eot(current_patch, n_eot)
                
                # Applica patch a tutti i target nell'immagine
                for person_bbox in bboxes:
                    patch_bbox = get_chest_bbox_proportional(person_bbox, img_size, img_size)
                    x1, y1, x2, y2 = patch_bbox
                    ph, pw = y2 - y1, x2 - x1
                    
                    if ph <= 0 or pw <= 0:
                        continue
                    
                    spatial_mask = self._build_spatial_mask(person_bbox, img_size, device)
                    if not spatial_mask.any():
                        continue
                    
                    # TACTICAL FILTER 2026: peso di rilevanza tattica in base
                    # all'altezza del bbox (0 sotto 60px, rampa 60-150px, 1
                    # sopra 150px). La patch viene comunque posizionata sul
                    # canvas per TUTTI i target (realismo fisico invariato,
                    # canvas sotto non tocco); solo il contributo alla LOSS
                    # viene escluso/attenuato per i target fuori scenario.
                    target_height = float(person_bbox[3] - person_bbox[1])
                    tactical_weight = self._tactical_weight(target_height)
                    if tactical_weight > 0:
                        global_spatial_mask_list.append(spatial_mask)
                        global_weight_list.append(spatial_mask.float() * tactical_weight)
                    
                    # Resize patch per bbox specifica
                    trans_rgb_res = F.interpolate(
                        trans_rgb, (ph, pw),
                        mode="bilinear",
                        align_corners=False
                    )
                    trans_mask_res = torch.clamp(
                        F.interpolate(trans_mask, (ph, pw), mode="bilinear", align_corners=False),
                        0.0, 1.0
                    )
                    
                    # Padding per posizionare patch nell'immagine
                    pad_left, pad_right = x1, img_size - x2
                    pad_top, pad_bottom = y1, img_size - y2
                    
                    canvas_rgb = F.pad(
                        trans_rgb_res,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        value=0.0
                    )
                    canvas_mask = F.pad(
                        trans_mask_res,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        value=0.0
                    )
                    
                    # Composizione alpha blending
                    global_canvas_rgb = (
                        global_canvas_rgb * (1 - canvas_mask) +
                        canvas_rgb * canvas_mask
                    )
                    global_canvas_mask = (
                        global_canvas_mask + canvas_mask -
                        (global_canvas_mask * canvas_mask)
                    )
                    
                    targets_in_image += 1
                
                if targets_in_image == 0:
                    continue
                
                # TACTICAL FILTER 2026: se NESSUN target nel frame supera il
                # peso 0 (tutti < 60px), il frame contribuisce zero alla loss
                # -- skip automatico senza scartare l'immagine dal dataloader,
                # coerente con la richiesta. La patch resta comunque allenata
                # sul resto dei frame del batch/step successivo.
                if not global_weight_list:
                    continue
                
                # Combina spatial masks pesate di tutti i target (max per
                # cella, per il raro caso di sovrapposizione tra bbox)
                global_cell_weight = torch.stack(global_weight_list).amax(dim=0)
                global_spatial_mask = global_cell_weight > 0
                
                # .contiguous() finale: seconda protezione per lo stesso
                # bug di stride, nel caso il blending non lo risolva.
                adv_batch = (
                    img_t.unsqueeze(0).expand(n_eot, -1, -1, -1) * (1 - global_canvas_mask) +
                    global_canvas_rgb * global_canvas_mask
                ).contiguous()
                
                # Calcola loss asintotica (pesata tatticamente)
                loss = self._asymptotic_loss(
                    torch_model, adv_batch, global_spatial_mask,
                    cell_weights=global_cell_weight
                )
                
                # Backward (accumula gradienti)
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_counter += 1
                
                # Norma del gradiente della SOLA loss principale, prima che la TV
                # aggiunga il suo contributo (calcolata ad ogni step raw, ma loggata
                # solo al momento dell'update per restare allineata alle altre serie)
                main_loss_grad_norm = self.patch_logits.grad.norm().item() if self.patch_logits.grad is not None else 0.0
                
                # Step optimizer solo dopo accumulation_steps
                if accumulation_counter >= GRADIENT_ACCUMULATION_STEPS:
                    # TV Loss (grafo pulito)
                    current_patch_tv = torch.sigmoid(self.patch_logits)
                    tv = self._tv_loss(current_patch_tv)
                    (TV_WEIGHT * tv).backward()
                    
                    # Gradient clipping per stabilità — la funzione ritorna la norma
                    # PRIMA del clipping: la usiamo per capire se il segnale che arriva
                    # alla patch è sano (es. 0.1-10) o quasi morto (es. 1e-6)
                    grad_norm = torch.nn.utils.clip_grad_norm_([self.patch_logits], max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # Reset accumulation
                    accumulated_loss /= accumulation_counter
                    loss_history.append(accumulated_loss)
                    
                    # Log metriche
                    current_lr = optimizer.param_groups[0]['lr']
                    self.training_metrics["timestamp"].append(datetime.now().isoformat())
                    self.training_metrics["step"].append(step)
                    self.training_metrics["loss"].append(accumulated_loss)
                    self.training_metrics["tv_loss"].append(tv.item())
                    self.training_metrics["learning_rate"].append(current_lr)
                    self.training_metrics.setdefault("grad_norm", []).append(float(grad_norm))
                    self.training_metrics.setdefault("main_loss_grad_norm", []).append(main_loss_grad_norm)
                    
                    # Early stopping check (media mobile)
                    if len(loss_history) >= EARLY_STOPPING_WINDOW:
                        recent_loss = np.mean(loss_history[-EARLY_STOPPING_WINDOW:])
                        
                        if recent_loss < best_loss - 1e-4:
                            best_loss = recent_loss
                            best_step = step
                            steps_without_improvement = 0
                            self._save_checkpoint(step, is_best=True)
                        else:
                            steps_without_improvement += 1
                        
                        if steps_without_improvement >= EARLY_STOPPING_PATIENCE:
                            if verbose:
                                print(f"\n[Early Stopping] Nessuno miglioramento per {EARLY_STOPPING_PATIENCE} step")
                                print(f"[Early Stopping] Best step: {best_step}, Best loss: {best_loss:.6f}")
                            break
                    
                    # Checkpoint periodico
                    if step % CHECKPOINT_EVERY == 0 and step > 0:
                        self._save_checkpoint(step)
                        self._save_metrics()
                    
                    # Telemetria visiva
                    if verbose and (step % 20 == 0 or step == int(n_steps) - 1):
                        with torch.no_grad():
                            img0 = torch.from_numpy(
                                np.array(imgs_pil[0]).astype(np.float32) / 255.0
                            ).permute(2, 0, 1).to(device)
                            pb = get_chest_bbox_proportional(bboxes_list[0][0], img_size, img_size)
                            chk_img = img0.clone()
                            
                            eval_patch = torch.sigmoid(self.patch_logits).detach()
                            p_res = F.interpolate(
                                eval_patch.unsqueeze(0),
                                (pb[3]-pb[1], pb[2]-pb[0]),
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0)
                            chk_img[:, pb[1]:pb[3], pb[0]:pb[2]] = p_res
                            
                            pil = Image.fromarray(
                                (chk_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            )
                            conf_now = self._get_person_conf(model(pil, verbose=False))
                            conf_history.append((step, conf_now))
                            self.training_metrics["confidence"].append(conf_now)
                        
                        print(
                            f"  [Step {step:4d}] Loss={accumulated_loss:.4f} | "
                            f"TV={tv.item():.4f} | YOLO Conf={conf_now:.4f} | "
                            f"LR={current_lr:.5f} | GradNorm(tot)={float(grad_norm):.6f} | "
                            f"GradNorm(loss)={main_loss_grad_norm:.6f}"
                        )
                    
                    # Reset counter
                    accumulated_loss = 0.0
                    accumulation_counter = 0
                
                # Cleanup memoria (importante per MPS)
                del img_t, current_patch, trans_rgb, trans_mask
                del global_canvas_rgb, global_canvas_mask, adv_batch
                if targets_in_image > 0:
                    del global_spatial_mask
                torch.mps.empty_cache() if device == "mps" else None
                
        except KeyboardInterrupt:
            # Salvataggio emergenza su interruzione
            print("\n[Interrupt] Salvataggio emergenza patch corrente...")
            self._save_checkpoint(step, is_best=True)
            self._save_metrics()
            print(f"[Interrupt] Patch salvata in {BEST_PATCH_FILE}")
        
        # Salva metriche finali
        self._save_metrics()
        
        return {
            "patch": torch.sigmoid(self.patch_logits).detach().cpu(),
            "loss_history": loss_history,
            "conf_history": conf_history,
            "best_step": best_step,
            "best_loss": best_loss
        }