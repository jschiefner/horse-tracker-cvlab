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
    from yolo_box_detector import BoxDetector
    box_detector = BoxDetector()
from kalman import Kalman2D

# %% class

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height

class VideoMaker():
    def __init__(self, cap, out):
        self.cap = cap
        self.out = out
        self.tracker = None
        self.smooth = None
        self.kalman = None
        self.kalman2d = None
        self.height_history = None
        self.tracked = 0
        
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        if len(boxes) == 0:
            return False, None
        else:
            left, top, right, bottom = boxes[0]
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            self.tracker = cv2.TrackerGOTURN_create()
            box = (left, top, right-left, bottom-top)
            ret = self.tracker.init(frame, box)
            return ret, (left, top, right, bottom)
            
    def track(self, frame):
        self.tracked += 1
        ret, box = self.tracker.update(frame)
        if ret:
            left, top = (int(box[0]), int(box[1]))
            right, bottom = (int(box[0] + box[2]), int(box[1] + box[3]))
            if draw: cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            return (left, top, right, bottom)
        else:
            raise RuntimeError('tracker error!')
            
    def calculate_height(self, box):
        _, top, _, bottom = box
        new = bottom - top
        if self.height_history is None:
            self.height_history = np.full(50, new, dtype=int)
        self.height_history[:-1] = self.height_history[1:]
        self.height_history[-1] = new
        return int(np.mean(self.height_history))
        
    def zoom(self, frame, center, height):
        height = height + 300 # todo: better value!
        width = height * ratio
        x, y = int(center[0]), int(center[1])
        dist_x = int(width // 2)
        dist_y = int(height // 2)
        left = x - dist_x; right = x + dist_x
        top = y - dist_y; bottom = y + dist_y
        # todo: make this more efficient (np.clip)
        if left < 0: left = 0
        if right < 0: right = 0
        if bottom < 0: bottom = 0
        if top < 0: top = 0
        cropped = frame[top:bottom, left:right]
        resized = cv2.resize(cropped, (frame_width, frame_height))
        return resized
            
    def draw_fixed(self, frame, center, height, color='red'):
        x, y = int(center[0]), int(center[1])
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
        while frame_number < frames:
            ret, frame = self.cap.read()
            if ret:
                if frame_number % 10 == 0: print('')
                print(f'{frame_number+1}/{frames} ', end='')
                if self.tracker is None:
                    ret, box = self.detect(frame)
                    if frame_number == 0: self.kalman2d = Kalman2D(box)
                    if not ret:
                        if self.tracker is None:
                            print('skipped frame {} because the tracker could not be initialized'.format(frame_number))
                            frame_number += 1
                            continue
                        box = self.track(frame)    
                else:
                    box = self.track(frame)
                x,y = self.kalman2d.predict()
                height = self.calculate_height(box)
                self.kalman2d.correct(box)
                frame = self.zoom(frame, (x,y), height)
                self.out.write(frame)
                frame_number += 1
                bar.next()
            
        # cleanup
        self.cap.release()
        self.out.release()
        bar.finish()
        print(f'\nTracked {self.tracked} frames')
        print('Result saved to \033[92m{}\033[00m'.format(output))
        
# %% action

# if len(sys.argv) != 7: raise RuntimeError('Programm needs 6 arguments to run: file, output, skip, frames, interval, draw got {} argument(s).'.format(len(sys.argv)-1))
# _, file, output, skip, frames, interval, draw = sys.argv
file, output, skip, frames, interval, draw = 'data/videos/GP028294.MP4', 'out/kalman6.avi', '240', '250', '1', 'f'

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
try:
    maker = VideoMaker(cap, out)
    maker.create()
except KeyboardInterrupt:
    print(f'\nInterrupted. Closing both files and saving result to \033[92m{output}\033[00m, tracked {maker.tracked} frames')
    cap.release()
    out.release()
