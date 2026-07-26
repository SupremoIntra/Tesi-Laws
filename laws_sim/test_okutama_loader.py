"""
Verifica rapida di OkutamaLoader sui dati reali.
Va copiato in laws_sim/ (root del progetto) ed eseguito da lì.

Uso:
    python test_okutama_loader.py --data data/okutama_train
"""
import argparse
import sys
sys.path.insert(0, "src")

from okutama_loader import OkutamaLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    loader = OkutamaLoader(args.data)
    print(f"\nTotale frame validi: {len(loader)}")

    for i in range(min(args.n_samples, len(loader))):
        img, bboxes = loader.get_sample(i)
        print(f"  sample {i}: img_size={img.size}, n_bbox={len(bboxes)}, bboxes={bboxes[:3]}")


if __name__ == "__main__":
    main()
