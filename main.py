import time
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join 
from collections import deque
from lane_detection import color_frame_pipeline
from car import load_cascade, detect_cars, draw_rectangles

# Constant: File containing the pre-trained Haar cascade for car detection
CASCADE_FILE = 'cars.xml'  

if __name__ == '__main__':

    resize_h, resize_w = 540, 960
    # adding the card model
    car_cascade = load_cascade(CASCADE_FILE)

    verbose = True

    # test on videos
    test_videos_dir = join('data', 'test_videos')
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
            

            # lines drawed 
            lines_frame = cv2.resize(frame, (resize_w, resize_h))
            img_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_gaussian_blur = cv2.GaussianBlur(img_gray_frame, (17,17), 0)
            img_edge_detection = cv2.Canny(img_gaussian_blur, threshold1=50, threshold2=80)

            frame_start_time = time.time()
            # Detect and annotate cars in the frame
            cars = detect_cars(frame, car_cascade)
            draw_rectangles(frame, cars)
            cv2.putText(frame, f'Detected Cars: {len(cars)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # final drawed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_w, resize_h))

            frame_buffer.append(frame)
            blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)

            cv2.imshow('video', img_edge_detection)
            cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()


