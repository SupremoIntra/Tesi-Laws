"""
Auto-Annotator per Custom Dataset (Domain Adaptation).
Converte foto raw in un dataset compatibile con VisDroneLoader.
"""

import os
import glob
from ultralytics import YOLO

def main():
    print("\n[1/3] Inizializzazione Auto-Annotatore...")
    
    # Definiamo la struttura delle cartelle
    base_dir = "custom_dataset"
    img_dir = os.path.join(base_dir, "images")
    ann_dir = os.path.join(base_dir, "annotations")
    
    # Creiamo la cartella annotations se non esiste
    os.makedirs(ann_dir, exist_ok=True)
    
    # Controlliamo che ci siano immagini
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        
    if not img_paths:
        print(f"[ERRORE] Nessuna immagine trovata in '{img_dir}'.")
        print("Assicurati di aver messo le tue foto (.jpg o .png) in quella cartella.")
        return
        
    print(f"      Trovate {len(img_paths)} immagini. Caricamento YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    print("\n[2/3] Generazione Annotazioni in formato VisDrone...")
    persone_totali = 0
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        txt_path = os.path.join(ann_dir, f"{stem}.txt")
        
        # Eseguiamo YOLO sull'immagine
        results = model(img_path, verbose=False)[0]
        
        annotazioni = []
        for box in results.boxes:
            # Classe 0 è "person" in COCO
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                
                # VisDrone format: bbox_left, bbox_top, bbox_width, bbox_height, score, category, trunc, occl
                width = x2 - x1
                height = y2 - y1
                left = x1
                top = y1
                
                # Category 1 = pedestrian in VisDrone
                # Formattiamo a interi per le coordinate, due decimali per lo score
                riga = f"{int(left)},{int(top)},{int(width)},{int(height)},{conf:.2f},1,0,0\n"
                annotazioni.append(riga)
                persone_totali += 1
                
        # Salviamo il file .txt anche se vuoto (segnala che non ci sono target)
        with open(txt_path, "w") as f:
            f.writelines(annotazioni)
            
        print(f"      - {filename}: {len(annotazioni)} persone rilevate.")

    print(f"\n[3/3] Operazione Completata!")
    print(f"      Dataset '{base_dir}' pronto per l'addestramento.")
    print(f"      Persone totali annotate: {persone_totali}")

if __name__ == "__main__":
    main()