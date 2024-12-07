import cv2
import time
from ultralytics import YOLO

def load_yolo_model(model_path):
    #Cargar el modelo YOLO
    return YOLO(model_path)

def detect_traffic_signs(frame, model):
    #Realiza la deteccion en el cuadro
    results = model.predict(frame)
    #Lista para almacenar las detecciones
    detections = []

    #Itera sobre las detecciones del primer resultado(imagen actual)
    for box in results[0].boxes:
        #Obtiene las coordenadas de la caja y la clase detectada
        xyxy = box.xyxy[0].cpu().numpy() #Coordenadas de la caja
        cls = int(box.cls[0].cpu().numpy()) #Clase del objeto
        conf = float(box.conf[0].cpu().numpy()) #Confianza de la deteccion

        detections.append((xyxy, cls, conf))


    return detections

def draw_detections(frame, detections, class_names):
    for detection in detections:
        xyxy, cls, conf = detection
        x1, y1, x2, y2 = map(int, xyxy) #Convierte coordenadas a enteros
        label = f"{class_names[cls]} {conf:.2f}"

        #Dibuja la caja y la etiqueta en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Load the Haar cascade classifier
def load_cascade(cascade_file):
    car_cascade = cv2.CascadeClassifier(cascade_file)
    if car_cascade.empty():
        raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
    print("Cascade classifier loaded successfully.")
    return car_cascade

def detect_cars(frame, car_cascade):
    """Detect cars in a video frame."""
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    return car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def draw_rectangles(frame, cars):
    """Draw rectangles around detected cars."""
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)