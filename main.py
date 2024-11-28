import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import color_frame_pipeline

if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    verbose = True

    # test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:

        print('Processing video: {}'.format(test_video))

        cap = cv2.VideoCapture(test_video)

        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if ret:
                # lines drawed 
                lines_frame = cv2.resize(color_frame, (resize_w, resize_h))
                img_gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                img_gaussian_blur = cv2.GaussianBlur(img_gray_frame, (17,17), 0)
                img_edge_detection = cv2.Canny(img_gaussian_blur, threshold1=50, threshold2=80)

                # final drawed
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, (resize_w, resize_h))

                frame_buffer.append(color_frame)

                # result of lines in frame
                # result_Canny = 
                # result of drawed in frame
                blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)

                cv2.imshow('video', img_edge_detection)
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


