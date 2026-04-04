from ultralytics import YOLO
import cv2
import cvzone # Libreria per disegnare i rettangoli e i testi in modo più semplice

cap = cv2.VideoCapture(0) # 0 per la webcam integrata
cap.set(3, 1280) #  width
cap.set(4, 720) # height

model = YOLO("../Yolo-Weights/yolov8l.pt") # Carica il modello YOLOv8 nano

while True:
    success, img = cap.read() # Legge un frame dalla webcam
    results = model(img, stream=True) # Esegui il rilevamento sull'immagine, stream=True per ottenere i risultati in tempo reale, più efficiente
    
    for r in results:
        boxes = r.boxes # Ottieni le bounding box
        for box in boxes:
            
            #PRIMO METODO OPENCV
            x1, y1, x2, y2 = box.xyxy[0] # Ottieni le coordinate della bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Converti le coordinate in interi (per disegnare i rettangoli con OpenCV)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Disegna un rettangolo intorno all'oggetto rilevato, do l'img, le coordinate del rettangolo, il colore (in questo caso rosa) e lo spessore del rettangolo
            
            
            #SECONDO METODO CVZONE
            w, h = x2 - x1, y2 - y1 # Calcola la larghezza e l'altezza del rettangolo
            
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255)) # Disegna un rettangolo con angoli arrotondati intorno all'oggetto rilevato, do l'img, le coordinate del rettangolo, la larghezza e l'altezza del rettangolo, la lunghezza degli angoli arrotondati, il raggio degli angoli arrotondati e il colore (in questo caso rosa)
    
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)