# %% imports
# %load_ext autoreload
# %autoreload 2
import sys
import warnings
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video_manager import VideoManager
from horse import Horse

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height
logger = logging.getLogger('horse')

# %% class

def find_closest_box(horse, boxes):
    distances = np.empty(len(boxes), dtype=np.float32)
    for index, box in enumerate(boxes):
        box_distance = horse.distance(box)
        distances[index] = box_distance
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    if min_distance > horse.allowed_distance():
        logger.info(f'allowed distance was not high enough ({min_distance}, {horse.allowed_distance()})')
        return None
    return boxes[min_index]
        
class Manager():
    def __init__(self, input, output, max_frames, skip, show, detector):
        self.video = VideoManager(input, output, max_frames, skip, show)
        self.horse = None
        self.detector = detector
        
    def match(self, frame, boxes, scores):
        # if no horse and no boxes: return
        # if no horse and boxes: assign first box
        # if horse but no boxes: track
        # if horse and boxes: find closest box
        logger.info(f'detected boxes: {boxes}')
        if self.horse is None and len(boxes) == 0: return
        elif self.horse is None and len(boxes) > 0:
            max_index = np.argmax(scores)
            self.horse = Horse(boxes[max_index], 1)
        elif self.horse is not None and len(boxes) == 0:
            self.horse.track(frame)
        elif self.horse is not None and len(boxes) > 0:
            box = find_closest_box(self.horse, boxes)
            if box is None:
                #self.horse.track() # old
                self.horse.track(frame)
            else:
                self.horse.detect(box)
        
    def initialize(self):
        raw = self.video.read()
        frame = raw.copy()
        smooth = raw.copy()
        boxes, scores = self.detector.detect_boxes(frame)
        self.match(frame, boxes, scores)
        if self.horse is not None:
            self.horse.draw(frame)
            self.horse.draw_smooth(smooth)
        horses = [] if self.horse is None else [self.horse]
        self.video.write(raw, frame, smooth, horses)
    
    def update(self):
        raw = self.video.read()
        frame = raw.copy()
        smooth = raw.copy()
        if self.horse is not None:
            self.horse.updated = False
            self.horse.last_detected += 1
        boxes, scores = self.detector.detect_boxes(frame)
        self.match(frame, boxes, scores)
        if self.horse is not None:
            self.horse.update(frame, [])
            self.horse.draw(frame)
            self.horse.draw_smooth(smooth)
        if self.horse is not None and self.horse.gone():
            logger.info('horse is gone')
            self.horse = None
        horses = [] if self.horse is None else [self.horse]
        self.video.write(raw, frame, smooth, horses)
