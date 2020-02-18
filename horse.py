import cv2
import random
import math
import numpy as np

class Horse():    
    horse_number = 1
    
    def __init__(self, box, color=None):
        self.box = box
        self.last_detected = 0
        self.tracker = None
        self.kalman2d = None
        self.height_history = None
        self.direction_history = np.zeros(5, dtype=np.float32)
        self.tracked = 0
        if color is None:
            self.color = (0, random.randint(0,255), random.randint(0,255))
        else:
            self.color = color
        self.tracking_color = (255, random.randint(0,50), random.randint(0,50))
        self.number = Horse.horse_number
        Horse.horse_number += 1
        
    def start_tracker(self, frame):
        # self.tracker = cv2.TrackerGOTURN_create()
        self.tracker = cv2.TrackerCSRT_create()
        left, top, right, bottom = self.box
        self.tracker.init(frame, (left, top, right-left, bottom-top))
        
    def track(self, frame):
        print(f'track horse {self.number}')
        _, box = self.tracker.update(frame)
        left, top = (int(box[0]), int(box[1]))
        right, bottom = (int(box[0] + box[2]), int(box[1] + box[3]))
        box = left, top, right, bottom
        self.update_direction(box)
        self.box = box
        
    def tracking(self):
        return self.tracker is not None
        
    def update_direction(self, box):
        new_x, _ = self.center(box)
        old_x, _ = self.center()
        movement = new_x - old_x
        print(new_x, old_x, movement)
        self.direction_history[:-1] = self.direction_history[1:]
        self.direction_history[-1] = movement
        
    def direction(self):
        mean = np.mean(self.direction_history)
        if mean > 0:
            return 1
        else:
            return -1
            
    def offset(self, box):
        horse_x, _ = self.center()
        box_x, _ = self.center(box)
        if box_x - horse_x > 0:
            return 1
        else:
            return -1
            
    def mean_direction(self):
        return np.mean(self.direction_history)
        
    def center(self, box=None):
        if box is None: box = self.box
        left, top, right, bottom = box
        x = np.mean((left, right)).astype(int)
        y = np.mean((top, bottom)).astype(int)
        return (x,y)
        
    def detected(self, box):
        print(f'detected horse {self.number}')
        self.update_direction(box)
        x_old, _ = self.center()
        x_new, _ = self.center(box)
        self.last_detected = 0
        self.tracker = None
        self.box = box
        
    def intersect(self, box):
        x, y = self.center()
        left, top, right, bottom = box
        horse_in_box = left <= x and right >= x and top <= y and bottom >= y
        x, y = self.center(box)
        left, top, right, bottom = self.box
        box_in_horse = left <= x and right >= x and top <= y and bottom >= y
        return horse_in_box or box_in_horse

    def height(self):
        _, top, _, bottom = self.box
        return bottom - top
        
    def distance(self, box):
        p1 = self.center()
        p2 = self.center(box)
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance
        
    def allowed_distance(self):
        factor = 50
        return self.last_detected * factor + factor
        
    def gone(self):
        return self.last_detected >= 100
        
    def draw(self, frame):
        left, top, right, bottom = self.box
        color = self.color if self.tracker is None else self.tracking_color
        center = self.center()
        cv2.rectangle(frame, (left, top), (right, bottom), color, 5)
        cv2.putText(frame, str(self.number), center, cv2.FONT_HERSHEY_TRIPLEX, 2, color)
