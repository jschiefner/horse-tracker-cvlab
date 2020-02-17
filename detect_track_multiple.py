# %% imports

import sys
import warnings
import math
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
from kalman import Kalman2D
from video_manager import VideoManager
import random

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height

# %% classes

class Horse():    
    def __init__(self, box):
        self.box = box
        self.last_detected = 0
        self.tracker = cv2.TrackerGOTURN_create()
        self.kalman2d = None
        self.height_history = None
        self.direction = 0
        self.tracked = 0
        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        
    # def startTracker(self):
    #     pass

    def center(self, box=None):
        if box is None: box = self.box
        left, top, right, bottom = box
        x = np.mean((left, right)).astype(int)
        y = np.mean((top, bottom)).astype(int)
        return (x,y)
        
    def detected(self, box):
        x_old, _ = self.center()
        x_new, _ = self.center(box)
        if x_new - x_old < 0: self.direction = -1
        else: self.direction = 1
        self.box = box
        
    def height(self):
        _, top, _, bottom = self.box
        return top - bottom
        
    def distance(self, box):
        p1 = self.center()
        p2 = self.center(box)
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance
        
    def draw(self, frame):
        left, top, right, bottom = self.box
        cv2.rectangle(frame, (left, top), (right, bottom), self.color, 5)

class Manager():
    def __init__(self, input, output, max_frames=25, skip=0):
        self.video = VideoManager(input, output, max_frames, skip)
        self.horses = np.array([], dtype=Horse)
        
    def addHorse(self, box):
        horse = Horse(box)
        self.horses = np.append(self.horses, horse)
    
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        relevant_boxes = []
        for index in range(len(boxes)):
            # todo: find appropriate value for low score
            if scores[index] > 0.1: relevant_boxes.append(boxes[index])
        return np.array(relevant_boxes)
        
    def match(self, detected):
        horses = self.horses.copy()
        for horse in horses:
            if len(detected) > 0:
                distances = np.empty(len(detected))
                for index, box in enumerate(detected):
                    distances[index] = horse.distance(box)
                min_index = distances.argmin()
                min_box = detected[min_index]
                horse.detected(min_box)
                detected = np.delete(detected, min_index, axis=0)
            # else:
            #     horse.startTracker()
        for box in detected:
            self.addHorse(box)
                
    def initialize(self):
        frame = self.video.read()
        detected = self.detect(frame)
        self.match(detected)
        for horse in self.horses:
            horse.draw(frame)
        self.video.write(frame)
        
    def update(self):
        frame = self.video.read()
        for horse in self.horses:
            horse.last_detected -= 1
        detected = self.detect(frame)
        self.match(detected)
        for horse in self.horses:
            horse.draw(frame)
        self.video.write(frame)

# %% action
skip = 5*23
frames = 8*23
manager = Manager('data/videos/GP038291.MP4', 'out/multiple1.avi', max_frames=frames, skip=skip)
manager.initialize()
for i in range(frames-1):
    try:
        manager.update()
    except KeyboardInterrupt:
        break
manager.video.close()