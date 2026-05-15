from visdrone_loader import VisDroneLoader
from PIL import ImageDraw

DATASET_PATH = "/Users/intra/visdrone" 

def main():
    print("Inizializzazione loader...")
    try:
        loader = VisDroneLoader(DATASET_PATH)
        loader.summary()
        
        # Prendiamo il primo frame valido
        img, bboxes = loader.get_sample(0)
        
        print(f"Trovati {len(bboxes)} pedoni. Disegno i Bounding Box...")
        
        # Disegniamo i rettangoli rossi sull'immagine
        draw = ImageDraw.Draw(img)
        for box in bboxes:
            # box è (x1, y1, x2, y2)
            draw.rectangle(box, outline="red", width=3)
            
        # Mostra l'immagine a schermo
        img.show()
        print("Testo rettangoli sulle persone???")
        
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()