# %% imports
# %load_ext autoreload
# %autoreload 2
import sys
import warnings
import logging
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
from horse import Horse

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height

logger = logging.getLogger('horse')
handler = logging.FileHandler('out/log.txt', 'a')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# %% class

# helper functions

distance_threshold = 300

def optimal(horse, detected):
    optimal_index = None
    optimal_distance = np.inf
    horse_direction = horse.direction()
    for index, box in enumerate(detected):
        distance = horse.distance(box)
        direction = horse.offset(box)
        mean_direction = horse.mean_direction()
        switch = 5
        logger.info(['eq dir:', direction, horse_direction, direction == horse_direction])
        logger.info(['mean', mean_direction, mean_direction > -switch and mean_direction < switch])
        logger.info(horse.movement_history)
        logger.info(['distance', distance, optimal_distance, distance < optimal_distance])
        if (direction == horse_direction or (mean_direction > -switch and mean_direction < switch) or distance <= distance_threshold) and distance < optimal_distance:
            optimal_distance = distance
            optimal_index = index
    return optimal_index, optimal_distance

# class
class Manager():
    def __init__(self, input, output, max_frames=25, skip=0):
        self.video = VideoManager(input, output, max_frames, skip)
        self.horses = np.array([], dtype=Horse)
        
    def addHorse(self, box):
        horse = Horse(box)
        logger.info(f'spawn horse {horse.number}')
        self.horses = np.append(self.horses, horse)
        
    def removeHorse(self, horse):
        logger.info(f'remove horse {horse.number}')
        self.horses = self.horses[self.horses != horse]
    
    def detect(self, frame):
        image = Image.fromarray(frame)
        boxes, scores = box_detector.detect_boxes(image)
        logger.info(['scores:', scores])
        relevant_boxes = []
        for index in range(len(boxes)):
            # todo: find appropriate value for low score
            if scores[index] > 0.5: relevant_boxes.append(boxes[index])
        return np.array(relevant_boxes)
        
    def match(self, frame, detected):
        for horse in self.horses:
            if len(detected) > 0:
                index, distance = optimal(horse, detected)
                logger.info(f'horse {horse.number} optimal index: {index}, distance: {distance}, allowed_distance: {horse.allowed_distance()}, last_detected: {horse.last_detected}')
                if (index is not None) and distance <= horse.allowed_distance():
                    min_box = detected[index]
                    horse.detect(min_box)
                    detected = np.delete(detected, index, axis=0)
                else:
                    horse.track(frame)
            else:
                horse.track(frame)
        for box in detected:
            intersects = False
            for horse in self.horses:
                if horse.intersect(box): intersects = True
            if not intersects:
                self.addHorse(box)
            
    def initialize(self):
        frame = self.video.read()
        detected = self.detect(frame)
        self.match(frame, detected)
        for horse in self.horses:
            horse.draw(frame)
        self.video.write(frame)
        
    def update(self):
        frame = self.video.read()
        for horse in self.horses:
            horse.last_detected += 1
        detected = self.detect(frame)
        self.match(frame, detected)
        for horse in self.horses:
            if horse.gone():
                self.removeHorse(horse)
                continue
            horse.update(frame, self.horses)
            horse.draw(frame)
        self.video.write(frame)
    
# %% action
# skip = 13*23
input_file = 'data/videos/GP038291.MP4'; skip = 8*23 + 37
# input_file = 'data/videos/Nachlieferung/Handorf/ZOOM0004_0.MP4'; skip = (7*60+13)*23
# input_file = 'data/videos/Nachlieferung/Kirchhellen/ZOOM0001_1.MP4'; skip = 0
# input_file = 'data/videos/Nachlieferung/Handorf/ZOOM0004_0.MP4'
# input_file = 'data/videos/GP028294.MP4'; skip = 32*23 + 51 + 21
frames = 8*23
out = f'out/multiple{input()}.avi'
manager = Manager(input_file, out, max_frames=frames, skip=skip)
manager.initialize()
for i in range(frames-1):
    try:
        manager.update()
    except KeyboardInterrupt:
        break
manager.video.close()