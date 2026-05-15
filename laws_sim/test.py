import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from visdrone_loader import VisDroneLoader
from patch_optimizer import PatchOptimizer
import torch

# Sostituisci con il tuo percorso reale
DATASET_PATH = "/Users/intra/visdrone" 

def main():
    print("1. Caricamento Dataset...")
    loader = VisDroneLoader(DATASET_PATH)
    
    print("\n2. Inizializzazione Optimizer...")
    opt = PatchOptimizer()
    
    # Controllo device per rassicurarci sull'M4
    device = next(opt._get_model().model.parameters()).device
    print(f"-> YOLO sta girando su: {device}")
    
    print("\n3. Avvio Smoke Test (10 step, batch_size=1)...")
    try:
        # Facciamo girare solo 10 step con 1 immagine alla volta per non fondere il Mac
        res = opt.optimize_universal(
            loader=loader, 
            n_steps=10, 
            batch_size=1, 
            verbose=True
        )
        print("\n✅ TEST SUPERATO! Il loop di ottimizzazione universale gira senza crash.")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE IL LOOP: {e}")

if __name__ == "__main__":
    main()