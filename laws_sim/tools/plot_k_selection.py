"""
Scelta empirica di LOSS_TOP_K, metodo indicato dal relatore.

Per ogni frame del dataset (nessuna patch applicata: caratterizziamo il
comportamento NATURALE di YOLO sulle persone vere, per decidere quante
celle "contano" in una detection tipica) si estraggono le confidenze
grezze di tutte le 8400 celle (prima di NMS/soglia) e si ordinano una
sola volta per confidenza decrescente. Con una somma cumulativa si
ottengono TP/FP/TN/FN, a livello di CELLA, per ogni K da 1 a 8400 senza
rifare l'inferenza — il costo è dominato dalla singola passata YOLO sul
dataset, non dal ciclo su K.

Definizioni (livello cella, non livello frame — vedi motivazione nel
docs/thesis_notes.md, da confermare col relatore):
    TP(K) = celle-bersaglio dentro le top-K per confidenza
    FN(K) = celle-bersaglio fuori dalle top-K
    FP(K) = celle-sfondo dentro le top-K
    TN(K) = celle-sfondo fuori dalle top-K
    R1(K) = sensitivity = TP/(TP+FN)   -> sale con K
    R2(K) = specificity = TN/(TN+FP)   -> scende con K
    F1(K), geometric_mean_recall(K) = sqrt(R1*R2)

Uso:
    python tools/plot_k_selection.py --data data/visdrone_val --max-samples 300
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
import matplotlib.pyplot as plt
from ultralytics import YOLO

from visdrone_loader import VisDroneLoader
from patch_optimizer import PatchOptimizer
from config import PERSON_CLASS_ID, IMG_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--max-samples", type=int, default=300,
                         help="Frame da scandire (non serve l'intero dataset per una curva stabile)")
    parser.add_argument("--k-max-plot", type=int, default=300,
                         help="K massimo mostrato nei plot (i conteggi esatti restano disponibili fino a 8400)")
    parser.add_argument("--out-dir", type=str, default="outputs/metrics")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    loader = VisDroneLoader(args.data)
    model = YOLO(args.model)
    model.to(device)
    torch_model = model.model
    torch_model.eval()

    n_cells = None  # determinato dal primo forward (tipicamente 8400 per YOLOv8 @ 640)
    tp_cum = fn_cum = fp_cum_cell = tn_cum_cell = None  # cumulativi per R1/F1 (livello cella, dentro il bersaglio)

    # OPZIONE 4: soglia implicita tau(K), che collega Approccio 1 e
    # Approccio 2 come indicato dalla nota ("uso quindi due approcci").
    # Su ogni frame POSITIVO, il valore di confidenza in posizione K nel
    # ranking (K-esimo piu' alto) e' la soglia minima che rende quella
    # cella parte del top-K. Aggregando (mediana) questi valori su tutti
    # i frame positivi si ottiene tau(K): una soglia equivalente ad
    # Approccio 1, ma derivata empiricamente da Approccio 2. Applicata ai
    # frame NEGATIVI da' un R2(K) che scende con K in modo naturale
    # (tau(K) e' monotona non-crescente: il decimo valore piu' alto e'
    # sempre <= del primo), risolvendo il problema delle opzioni 1-3.
    positive_sorted_confs = []  # lista di array (n_cells,) ordinati desc, uno per frame positivo
    negative_max_confs = []     # un valore per frame negativo

    indices = list(range(len(loader)))[:args.max_samples]
    n_used = 0
    n_skipped_no_target = 0

    print(f"Scansione di {len(indices)} frame (nessuna patch — comportamento naturale di YOLO)...")
    with torch.no_grad():
        for idx in indices:
            img_pil, gt_bboxes = loader.get_sample(idx)
            valid_bboxes = [b for b in gt_bboxes if (b[3] - b[1]) >= 60]

            img_t = torch.from_numpy(
                np.array(img_pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

            raw = torch_model(img_t)
            preds = raw[0] if isinstance(raw, (tuple, list)) else raw
            person_scores = torch.sigmoid(preds[0, 4 + PERSON_CLASS_ID, :]).float().cpu().numpy()  # (n_cells,)

            if n_cells is None:
                n_cells = person_scores.shape[0]
                tp_cum = np.zeros(n_cells, dtype=np.int64)
                fn_cum = np.zeros(n_cells, dtype=np.int64)
                fp_cum_cell = np.zeros(n_cells, dtype=np.int64)
                tn_cum_cell = np.zeros(n_cells, dtype=np.int64)

            # Maschera bersaglio: OR di tutte le persone valide nel frame
            target_mask = np.zeros(n_cells, dtype=bool)
            for bbox in valid_bboxes:
                m = PatchOptimizer._build_spatial_mask(bbox, IMG_SIZE, device="cpu").numpy()
                target_mask |= m

            n_target = int(target_mask.sum())

            if n_target == 0:
                # Frame SENZA bersaglio valido: contribuisce a R2 tramite
                # la sua confidenza massima, confrontata dopo con tau(K).
                negative_max_confs.append(float(person_scores.max()))
                n_skipped_no_target += 1  # non entra nel conteggio R1/F1 cell-level
                continue

            n_background = n_cells - n_target

            # Ordina le celle per confidenza decrescente UNA volta
            order = np.argsort(-person_scores)
            target_sorted = target_mask[order]  # True se la cella al rango i e' bersaglio
            sorted_confs = person_scores[order]  # per tau(K): valore di confidenza al rango K

            # Cumulativa: quante celle-bersaglio sono tra le prime K (per ogni K)
            target_in_topk = np.cumsum(target_sorted)  # TP(K) per K=1..n_cells
            k_range = np.arange(1, n_cells + 1)
            background_in_topk = k_range - target_in_topk  # FP(K), solo su questo frame positivo

            tp_cum += target_in_topk
            fn_cum += (n_target - target_in_topk)
            fp_cum_cell += background_in_topk
            tn_cum_cell += (n_background - background_in_topk)
            positive_sorted_confs.append(sorted_confs)

            n_used += 1

    if n_used == 0:
        print("Nessun frame valido scandito.")
        return

    # tau(K): mediana, sui frame positivi, del valore di confidenza al
    # rango K (monotona non-crescente per costruzione -- il decimo
    # valore piu' alto e' sempre <= del primo, in ogni singolo frame).
    pos_matrix = np.stack(positive_sorted_confs, axis=0)  # (n_frame_positivi, n_cells)
    tau_K = np.median(pos_matrix, axis=0)                  # (n_cells,)

    # R2(K): frazione di frame negativi con confidenza massima SOTTO tau(K)
    neg_max = np.array(negative_max_confs) if negative_max_confs else np.array([1.0])
    neg_max_sorted = np.sort(neg_max)
    n_below = np.searchsorted(neg_max_sorted, tau_K, side="left")  # per ogni K, quanti negativi < tau(K)
    R2 = n_below / max(len(neg_max), 1)

    print(f"Frame utilizzati (positivi, cell-level R1/F1): {n_used} | "
          f"frame negativi (per tau(K)/R2): {len(negative_max_confs)} | celle per frame: {n_cells}")

    eps = 1e-9
    K = np.arange(1, n_cells + 1)
    R1 = tp_cum / np.maximum(tp_cum + fn_cum, eps)          # sensitivity, cell-level, varia con K
    precision = tp_cum / np.maximum(tp_cum + fp_cum_cell, eps)
    f1 = 2 * precision * R1 / np.maximum(precision + R1, eps)
    # R2 gia' calcolata sopra (tau(K) derivata dai frame positivi, applicata ai negativi)
    gmean = np.sqrt(np.clip(R1, 0, 1) * np.clip(R2, 0, 1))

    k_best_gmean = int(K[np.argmax(gmean)])
    k_best_f1 = int(K[np.argmax(f1)])
    print(f"\nK che massimizza la media geometrica sqrt(R1*R2): {k_best_gmean} (valore={gmean.max():.4f})")
    print(f"K che massimizza F1: {k_best_f1} (valore={f1.max():.4f})")
    print(f"R2 a K=1: {R2[0]:.4f} | R2 a K={min(300,n_cells)}: {R2[min(300,n_cells)-1]:.4f} | R2 a K={n_cells}: {R2[-1]:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "k_selection_raw.npz"),
        K=K, R1=R1, R2=R2, F1=f1, gmean=gmean
    )

    kmax = min(args.k_max_plot, n_cells)
    sl = slice(0, kmax)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(K[sl], f1[sl], color="#2980b9")
    axes[0, 0].axvline(k_best_f1, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set_title("F1 vs K")
    axes[0, 0].set_xlabel("K (top-k celle)")
    axes[0, 0].set_ylabel("F1-Score")

    axes[0, 1].plot(K[sl], gmean[sl], color="#8e44ad")
    axes[0, 1].axvline(k_best_gmean, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title(r"$\sqrt{R1 \times R2}$ vs K")
    axes[0, 1].set_xlabel("K (top-k celle)")
    axes[0, 1].set_ylabel("Media geometrica")

    axes[1, 0].plot(K[sl], R1[sl], color="#27ae60")
    axes[1, 0].set_title("R1 (sensitivity) vs K")
    axes[1, 0].set_xlabel("K (top-k celle)")
    axes[1, 0].set_ylabel("R1")

    axes[1, 1].plot(K[sl], R2[sl], color="#c0392b")
    axes[1, 1].set_title("R2 (specificity) vs K")
    axes[1, 1].set_xlabel("K (top-k celle)")
    axes[1, 1].set_ylabel("R2")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "k_selection_plots.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot salvati in: {out_path}")
    print(f"Dati grezzi (tutti i K fino a {n_cells}) salvati in: k_selection_raw.npz")


if __name__ == "__main__":
    main()
