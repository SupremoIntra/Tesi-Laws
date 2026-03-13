from ultralytics import YOLO
import cv2

# 1. Carica il modello base
model = YOLO("Yolo-Weights/yolov8n.pt") 

# 2. Fai l'inferenza SENZA show=True. YOLO fa solo matematica, niente grafica.
results = model("carro.jpg")  

# 3. Estrai l'immagine con i rettangoli disegnati sopra
# results è una lista. Prendiamo il primo risultato [0] e usiamo .plot() 
# per farci restituire la matrice dell'immagine pronta per OpenCV.
annotated_frame = results[0].plot()

# 4. Usiamo OpenCV "puro" per creare la finestra e mostrare la matrice
cv2.imshow("La Visuale del Drone", annotated_frame)

# 5. Ora OpenCV ha il controllo totale: aspetta all'infinito
cv2.waitKey(0)

# 6. Buona pratica su Mac: quando premi un tasto, chiudi pulito
cv2.destroyAllWindows()
