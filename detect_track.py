# %% imports 

import sys
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from progress.bar import Bar
from progress.spinner import Spinner
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from box_detector import BoxDetector
    box_detector = BoxDetector()

# %% class

frame_width = 3840
frame_height = 2160

class VideoMaker():
    def __init__(self, cap, out):
        self.cap = cap
        self.out = out
        self.old_zoom = None
        self.tracker = None
        self.zoom_history = None
        self.smooth = None
        
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        if len(boxes) == 0:
            return False, None
        else:
            left, top, right, bottom = boxes[0]
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 10)
            self.tracker = cv2.TrackerGOTURN_create()
            bbox = (left, top, right-left, bottom-top)
            ret = self.tracker.init(frame, bbox)
            return ret, (left, top, right, bottom)
            
    def track(self, frame):
        ret, bbox = self.tracker.update(frame)
        if ret:
            left, top = (int(bbox[0]), int(bbox[1]))
            right, bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 10)
            return (left, top, right, bottom)
        else:
            raise RuntimeError('tracker error!')
            
    # def initialize_smoothing(self, frame, box):
    #     left, top, right, bottom = box
    #     x = np.mean((left, right)).astype(np.int)
    #     y = np.mean((top, bottom)).astype(np.int)
    #     self.smooth = Smooth(x, y)
    # 
    # def predict_kalman(self, box):
    #     x, y = self.smooth.predict()
    #     x = int(x[0][0]); y = int(y[0][0])
    #     return x, y
    # 
    # def update_kalman(self, box):
    #     left, top, right, bottom = box
    #     x = np.mean((left, right)).astype(np.int)
    #     y = np.mean((top, bottom)).astype(np.int)
    #     x, y = self.smooth.update(x, y)
    #     x = int(x[0][0]); y = int(y[0][0])
    #     return x, y
        
    def create(self):
        if skip > 0:
            spinner = Spinner('Skipping {} Frames... '.format(skip))
            frame_number = 0
            while frame_number < skip:
                spinner.next()
                ret, _ = self.cap.read()
                if ret: frame_number += 1
            print('\nSkipped {} frames'.format(skip))

        # iterate remaining frames
        frame_number = 0
        tracker_errors = 0
        bar = Bar('Processing frames', max=frames)
        drawchange = 1 # remove
        while frame_number < frames:
            print(f'{frame_number+1}/{frames}')
            ret, frame = self.cap.read()
            if ret:
                if self.tracker is None or frame_number % interval == 0:
                    ret, box = self.detect(frame)
                    if frame_number == 0: self.kalman = Kalman(box)
                    if not ret:
                        if self.tracker is None:
                            print('skipped frame {} because the tracker could not be initialized'.format(frame_number))
                            continue
                        box = self.track(frame)
                    # x, y = self.predict_kalman(box)
                    # print(f'predicted: {x}, {y}')
                    # frame = cv2.drawMarker(frame, (x,y), (0, 255, 0))
                    # if frame_number % 5 == 0:
                    #     x, y = self.update_kalman(box)
                    #     print(f'updated: {x}, {y}')
                    prediction = self.kalman.predict()
                    w, h = 50, 50
                    frame = cv2.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (0,255,0),2)
                    self.kalman.correct(box)
                else:
                    box = self.track(frame)
                self.out.write(frame)
                frame_number += 1
                bar.next()
            
        # cleanup
        self.cap.release()
        self.out.release()
        bar.finish()
        print('Result saved to \033[92m{}\033[00m'.format(output))
        
# import cv2
# import numpy as np

class Kalman():
    def __init__(self, box):
        kalman = cv2.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)

        kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)

        kalman.processNoiseCov = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]],np.float32) * 0.03
        self.kalman = kalman
        for i in range(5):
            self.correct(box)
            self.predict()
    
    def correct(self, box):
        left, top, right, bottom = box
        x = np.mean((left, right))
        y = np.mean((top, bottom))
        center = np.array([np.float32(x), np.float32(y)], np.float32)
        self.kalman.correct(center)
        
    def predict(self):
        return self.kalman.predict()

# %% action

# if len(sys.argv) != 7: raise RuntimeError('Programm needs 6 arguments to run: file, output, skip, frames, interval, draw got {} argument(s).'.format(len(sys.argv)-1))
# _, file, output, skip, frames, interval, draw = sys.argv
file, output, skip, frames, interval, draw = 'data/videos/Nachlieferung/Herbern/ZOOM0002_0.mp4', 'out/kalman2.avi', '0', '25', '1', 't'

skip = int(skip)
frames = int(frames)
interval = int(interval)
draw = draw == 'True' or draw == 'true' or draw == 't'
print('Arguments: file: {}, output: {}, skip: {}, frames: {}, interval: {}, draw: {}'.format(file, output, skip, frames, interval, draw))

cap = cv2.VideoCapture(file)
if (cap.isOpened() == False): 
    print('Unable to read video feed')
    exit(1)
    
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(output, fourcc, 25, (frame_width, frame_height))

maker = VideoMaker(cap, out)
maker.create()