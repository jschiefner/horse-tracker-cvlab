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
from kalman import Kalman, Kalman2D

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
        self.kalman = None
        self.kalman2d = None
        
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        if len(boxes) == 0:
            return False, None
        else:
            left, top, right, bottom = boxes[0]
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            self.tracker = cv2.TrackerGOTURN_create()
            bbox = (left, top, right-left, bottom-top)
            ret = self.tracker.init(frame, bbox)
            return ret, (left, top, right, bottom)
            
    def track(self, frame):
        ret, bbox = self.tracker.update(frame)
        if ret:
            left, top = (int(bbox[0]), int(bbox[1]))
            right, bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            return (left, top, right, bottom)
        else:
            raise RuntimeError('tracker error!')
            
    def draw_fixed(self, frame, pos, height, color='red'):
        x, y  = int(pos[0]), int(pos[1])
        color = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0)}[color]
        half_height = height // 2
        cv2.rectangle(frame, (x-half_height,y-half_height), (x+half_height,y+half_height), color, 5)
    
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
            ret, frame = self.cap.read()
            if ret:
                print(f'{frame_number+1}/{frames}')
                if self.tracker is None or frame_number % interval == 0:
                    ret, box = self.detect(frame)
                    if frame_number == 0:
                        self.kalman2d = Kalman2D(box)
                        self.kalman = Kalman(box)
                    if not ret:
                        if self.tracker is None:
                            print('skipped frame {} because the tracker could not be initialized'.format(frame_number))
                            continue
                        box = self.track(frame)
                    
                    height = self.kalman.predict()
                    x,y = self.kalman2d.predict()
                    self.draw_fixed(frame, (x,y), height, color='green')
                    self.kalman.correct(box)
                    self.kalman2d.correct(box)
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
        
# %% action

# if len(sys.argv) != 7: raise RuntimeError('Programm needs 6 arguments to run: file, output, skip, frames, interval, draw got {} argument(s).'.format(len(sys.argv)-1))
# _, file, output, skip, frames, interval, draw = sys.argv
file, output, skip, frames, interval, draw = 'data/videos/GP028294.MP4', 'out/kalman4.avi', '0', '100', '1', 'true'

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