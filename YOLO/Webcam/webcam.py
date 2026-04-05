from ultralytics import YOLO
import cv2
import cvzone # Libreria per disegnare i rettangoli e i testi in modo più semplice

cap = cv2.VideoCapture(0) # 0 per la webcam integrata
cap.set(3, 1280) #  width
cap.set(4, 720) # height

model = YOLO("../Yolo-Weights/yolov8l.pt") # Carica il modello YOLOv8 

while True:
    success, img = cap.read() # Legge un frame dalla webcam
    results = model(img, stream=True) # Esegui il rilevamento sull'immagine, stream=True per ottenere i risultati in tempo reale, più efficiente
    
    for r in results:
        boxes = r.boxes # Ottieni le bounding box dai risultati
        for box in boxes:
            #usiamo CVzone (più easy di OpenCV)
            x1, y1, x2, y2 = box.xyxy[0] # Ottieni le coordinate della bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Converti le coordinate in interi (per disegnare i rettangoli con OpenCV)
            
            
            #calcola la confidenza
            conf = box.conf[0].item() # Ottieni la confidenza del rilevamento
            cls = box.cls[0].item() # Ottieni la classe dell'oggetto
            
            if conf >0.80 :
                 cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=30, rt=2, colorR=(0, 255, 0)) 
                 
                 cvzone.putTextRect(img, f"{model.names[int(cls)]} {conf:.2f}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3) # Disegna il nome della classe e la confidenza sopra la bounding box
                 
            else:
                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=30, rt=2, colorR=(0, 0, 255)) 
                
                cvzone.putTextRect(img, f"{model.names[int(cls)]} {conf:.2f}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3) # Disegna il nome della classe e la confidenza sopra la bounding box
                
            
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)