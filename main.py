import time
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join 
from collections import deque
from lane_detection import color_frame_pipeline
from car import load_yolo_model, detect_traffic_signs, draw_detections, draw_detections_cars

#Ruta al modelo YOLO
YOLO_MODEL_PATH = "./models/bestGermanDetection.pt"
TRAFFIC_LIGHT_MODEL_PATH = "./models/bestTrafficLights.pt"
CAR_MODEL_PATH = "./models/yolov8s.pt"

#Clases de se;ales de trafico
CLASS_NAMES = [
        'Speed limit (20km/h)',
        'Speed limit (30km/h)',
        'Speed limit (50km/h)',
        'Speed limit (60km/h)',
        'Speed limit (70km/h)',
        'Speed limit (80km/h)',
        'End of speed limit (80km/h)',
        'Speed limit (100km/h)',
        'Speed limit (120km/h)',
        'No passing',
        'No passing for vechiles over 3.5 metric tons',
        'Road Block',
        'Priority road',
        'Yield',
        'Stop',
        'No vehicles',
        'Vechiles over 3.5 metric tons prohibited',
        'No entry',
        'General caution',
        'Double curve',
        'Bumpy Road',
        'Slippery road',
        'Road narrows on the right',
        'Road Work',
        'Traffic Signals',
        'Pedestrians',
        'Children crossing',
        'Bicycles crossing',
        'Beware of ice/snow',
        'Wild animals crossing',
        'End of all speed and passing limits',
        'Turn right ahead',
        'Turn left ahead',
        'Ahead only',
        'Go straight or right',
        'Go straight or left',
        'Keep right',
        'Keep left',
        'Roundabout mandatory',
        'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons']

#Clases de semaforos
TRAFFIC_LIGHT_CLASSES = ['Green Light', 'Red Light', 'Yellow Light']

#Cargar modelos
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
traffic_light_model = load_yolo_model(TRAFFIC_LIGHT_MODEL_PATH)
car_model = load_yolo_model(CAR_MODEL_PATH)

if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    verbose = True

    # test on videos
    test_videos_dir = join("./data", 'testing_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    # process_video(test_videos[1], car_cascade)
    for test_video in test_videos:
        print('Processing video: {}'.format(test_video))
        cap = cv2.VideoCapture(test_video)

        if not cap.isOpened():
            raise IOError(f"Error opening video file {test_video}")

        # get basic video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_buffer = deque(maxlen=10)
        print(f"Video Info: {total_frames} frames, {fps} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # final drawed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_w, resize_h))

            frame_buffer.append(frame)
            blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)
            
            # Detect cars
            cars_detected = detect_traffic_signs(blend_frame, car_model)
            draw_detections_cars(blend_frame, cars_detected, 'car')

            #Detect traffic signs
            traffic_signs = detect_traffic_signs(blend_frame, yolo_model)
            draw_detections(blend_frame, traffic_signs, CLASS_NAMES)

            #Detect TrafficLights
            traffic_lights = detect_traffic_signs(blend_frame, traffic_light_model)
            draw_detections(blend_frame, traffic_lights, TRAFFIC_LIGHT_CLASSES)
            #Mostrar cuantos semaforos se detectan
            cv2.putText(blend_frame, f'Detected Traffic Lights: {len(traffic_lights)}', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

            #Mostrar cuantas se;ales se detectan
            cv2.putText(blend_frame, f'Detected Traffic Signs: {len(traffic_signs)}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

            # cv2.imshow('video', img_edge_detection)
            cv2.imshow('Final', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR) ), cv2.waitKey(1)
            # cv2.imshow('Object detected', blend_frame ), cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()


