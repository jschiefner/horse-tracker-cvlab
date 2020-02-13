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

break_back = 200
break_front = 500
back_height = 580
middle_height = 830
front_height = 1550
back_y = 1140
middle_y = 1180
front_y = 1300
frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height

class VideoMaker():
    def __init__(self, cap, out):
        self.cap = cap
        self.out = out
        self.old_zoom = None
        self.tracker = None
        self.zoom_history = None
        self.draw = draw # remove

    def shift_zoom_history(self, new):
        if self.zoom_history is None:
            self.old_zoom = new
            self.zoom_history = np.full(20, new, dtype=int)
        self.zoom_history[:-1] = self.zoom_history[1:]
        self.zoom_history[-1] = new

    def determine_zoom(self, new_zoom):
        unique = np.unique(self.zoom_history)
        if unique.size == 1:
            return new_zoom
        else:
            return self.old_zoom

    def crop_and_resize(self, frame, box):
        left, top, right, bottom = box
        x = left + (right-left) // 2
        height = bottom - top
        
        if height <= break_back: new_zoom = 0
        elif height <= break_front: new_zoom = 1
        else: new_zoom = 2
        
        self.shift_zoom_history(new_zoom)
        new_zoom = self.determine_zoom(new_zoom)
            
        if new_zoom == 0:
            top = back_y - back_height // 2
            bottom = back_y + back_height // 2
            width = back_height * ratio
            left = x - int(width // 2)
            right = x + int(width // 2)
        elif new_zoom == 1:
            top = middle_y - middle_height // 2
            bottom = middle_y + middle_height // 2
            width = middle_height * ratio
            left = x - int(width // 2)
            right = x + int(width // 2)
        elif new_zoom == 2:
            top = front_y - front_height // 2
            bottom = front_y + front_height // 2
            width = front_height * ratio
            left = x - int(width // 2)
            right = x + int(width // 2)
        else:
            raise RuntimeError('zoom factor {} not supported'.format(new_zoom))

        cropped = frame[top:bottom, left:right]
        resized = cv2.resize(cropped, (frame_width, frame_height))
        self.old_zoom = new_zoom
        return resized
        
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        if len(boxes) == 0:
            return False, None
        else:
            left, top, right, bottom = boxes[0]
            if self.draw: cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 10) # replace self.draw with draw
            self.tracker = cv2.TrackerGOTURN_create()
            bbox = (left, top, right-left, bottom-top)
            ret = self.tracker.init(frame, bbox)
            return ret, (left, top, right, bottom)
            
    def track(self, frame):
        ret, bbox = self.tracker.update(frame)
        if ret:
            left, top = (int(bbox[0]), int(bbox[1]))
            right, bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            if self.draw: cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 10) # replace self.draw with draw
            return (left, top, right, bottom)
        else:
            raise RuntimeError('tracker error!')
            
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
                if self.tracker is None or frame_number % interval == 0:
                    ret, box = self.detect(frame)
                    if not ret:
                        if self.tracker is None:
                            print('skipped frame {} because the tracker could not be initialized'.format(frame_number))
                            continue
                        box = self.track(frame)
                else:
                    box = self.track(frame)
                try:
                    frame = self.crop_and_resize(frame, box)
                    self.out.write(frame)
                    frame_number += 1
                except:
                    print('resize error')
                # alternate drawing # remove
                if (frame_number // 200) == drawchange:
                    self.draw = not self.draw
                    drawchange += 1 
                # alternate drawing end # remove
                bar.next()
            
        # cleanup
        self.cap.release()
        self.out.release()
        bar.finish()
        print('Result saved to \033[92m{}\033[00m'.format(output))

# %% action

if len(sys.argv) != 7: raise RuntimeError('Programm needs 6 arguments to run: file, output, skip, frames, interval, draw got {} argument(s).'.format(len(sys.argv)-1))
_, file, output, skip, frames, interval, draw = sys.argv
# file, output, skip, frames, interval, draw = 'GP028294.MP4', 'output.avi', '650', '150', '15', 'false'

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
out = cv2.VideoWriter(output, fourcc, 25, (frame_width,frame_height))

maker = VideoMaker(cap, out)
maker.create()