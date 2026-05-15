import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from visdrone_loader import VisDroneLoader
from simulator import evaluate_on_dataset

# Il tuo percorso
DATASET_PATH = "/Users/intra/visdrone"

def main():
    print("=== AVVIO VALUTAZIONE EMPIRICA VISDRONE (BASELINE) ===")
    loader = VisDroneLoader(DATASET_PATH)
    
    # Valutiamo i primi 100 frame per fare in fretta prima che tu debba scappare
    metrics = evaluate_on_dataset(
        loader=loader,
        patch_tensor=None,  # Nessuna patch -> Stiamo calcolando la Baseline pura
        conf_threshold=0.50,
        max_samples=100,
        verbose=True
    )
    
    print("\n" + "="*40)
    print("🎯 RISULTATI REALI (GROUND TRUTH / BASELINE)")
    print("="*40)
    print(f"F1-Score:  {metrics.f1:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"TP: {metrics.tp} | FP: {metrics.fp} | FN: {metrics.fn}")
    print("="*40)

if __name__ == "__main__":
    main()