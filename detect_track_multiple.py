# %% imports
# %load_ext autoreload
# %autoreload 2
import sys
import warnings
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from progress.spinner import Spinner
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from background_box_detector import BoxDetector 
from video_manager import VideoManager
from horse import Horse
from timeit import default_timer as timer

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

def find_closest_horse(horses, box):
    distance = np.inf
    optimal_index = None
    for index, horse in enumerate(horses):
        box_distance = horse.distance(box)
        if box_distance < distance:
            distance = box_distance
            optimal_index = index
    if optimal_index is None:
        logger.info('no optimal index found')
        return None
    horse = horses[optimal_index]
    if distance >= horse.allowed_distance():
        logger.info(f'allowed distance was not high enough ({distance}, {horse.allowed_distance()})')
        return None
    direction = horse.offset(box)
    horse_direction = horse.direction()
    if direction != horse_direction and distance >= distance_threshold:
        logger.info(f'direction was wrong (box: {direction}, horse: {horse_direction}) and distance was too big ({distance}/{distance_threshold})')
        return None
    logger.info(f'horse {horse.number} made it through')
    return horse
    
def list_diff(list1, list2):
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)
    return out

# class
class Manager():
    def __init__(self, input, output, max_frames=25, skip=0, show=True):
        self.video = VideoManager(input, output, max_frames, skip, show)
        self.horses = np.array([], dtype=Horse)
        
    def spawnHorse(self, box):
        global global_horse_number
        horse = Horse(box, global_horse_number)
        global_horse_number += 1
        logger.info(f'spawn horse {horse.number}')
        self.horses = np.append(self.horses, horse)
        return horse
        
    def removeHorse(self, horse):
        logger.info(f'remove horse {horse.number}')
        self.horses = self.horses[self.horses != horse]
    
    def detect(self, frame):
        boxes, scores = box_detector.detect_boxes(frame)
        relevant_boxes = []
        for index in range(len(boxes)):
            # todo: find appropriate value for low score
            if scores[index] > 0.3: relevant_boxes.append(boxes[index])
        return np.array(relevant_boxes)
                
    def match(self, frame, detected):
        detected_horses = []
        for index, box in enumerate(detected):
            intersects = False
            for horse in detected_horses:
                if horse.intersect(box): intersects = True
            if intersects: continue
            lone_horses = list_diff(self.horses, detected_horses)
            horse = find_closest_horse(lone_horses, box)
            if horse is None:
                horse = self.spawnHorse(box)
                detected_horses.append(horse)
            else:
                horse.detect(box)
            detected_horses.append(horse)
        lone_horses = list_diff(self.horses, detected_horses)
        for horse in lone_horses:
            horse.track(frame)
            
    def initialize(self):
        raw = self.video.read()
        frame = raw.copy()
        smooth = raw.copy()
        detected = self.detect(frame)
        self.match(frame, detected)
        for horse in self.horses:
            horse.draw(frame)
            horse.draw_smooth(smooth)
        self.video.write(raw, frame, smooth, self.horses)
        
    def update(self):
        raw = self.video.read()
        frame = raw.copy()
        smooth = raw.copy()
        for horse in self.horses:
            horse.updated = False
            horse.last_detected += 1
        detected = self.detect(frame)
        self.match(frame, detected)
        for horse in self.horses:
            if horse.gone():
                self.removeHorse(horse)
                continue
            horse.update(frame, self.horses)
            horse.draw(frame)
            horse.draw_smooth(smooth)
        self.video.write(raw, frame, smooth, self.horses)

# %% action
# skip = 13*23
input_file = 'data/videos/GP028294.MP4'; out = 'out/one_horse_background2'; skip = 0; frames = 125+375
# input_file = 'data/videos/Nachlieferung/Handorf/ZOOM0004_0.MP4'; skip = (7*60+13)*23
# input_file = 'data/videos/Nachlieferung/Kirchhellen/ZOOM0001_1.MP4'; skip = 0
# input_file = 'data/videos/Nachlieferung/Handorf/ZOOM0004_0.MP4'
# input_file = 'data/videos/GP028294.MP4'; skip = 32*23 + 51 + 21
# input_file = 'data/videos/Nachlieferung/Handorf/ZOOM0004_0.mp4'; skip = 3680; frames = 598
global_horse_number = 0
box_detector = BoxDetector()
# out = 'out/two_horses'
# _, input_file, out, frames, skip, = sys.argv
frames = int(frames)
skip = int(skip)
manager = Manager(input_file, out, max_frames=frames, skip=skip, show=False)
manager.initialize()
start = timer()
for i in range(frames-1):
    try:
        manager.update()
    except KeyboardInterrupt:
        break
end = timer()
elapsed_seconds = end-start; elapsed_minutes = elapsed_seconds / 60
logger.info(f'time elapsed: \033[92m{elapsed_seconds}\033[00m seconds => \033[92m{elapsed_minutes}\033[00m minutes.')
manager.video.close()