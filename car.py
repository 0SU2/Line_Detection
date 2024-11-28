import cv2
import time

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