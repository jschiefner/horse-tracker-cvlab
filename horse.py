import cv2
import random
import math
import numpy as np
from smooth import Smoother
import logging
logger = logging.getLogger('horse')

blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
yellow = (0,225,225)

def center(box):
    left, top, right, bottom = box
    x = np.mean((left, right)).astype(int)
    y = np.mean((top, bottom)).astype(int)
    return (x,y)
    
def fixed_box(x, y, height):
    half_height = height // 2
    left = x - half_height
    top = y - half_height
    right = x + half_height
    bottom = y + half_height
    return left, top, right, bottom
    
class Horse():        
    def __init__(self, box, number):
        self.box = box
        self.smooth_box = box
        self.last_detected = 0
        self.tracker = None
        self.height_history = None
        self.movement_history = np.zeros(10, dtype=np.float64)
        self.status = 'detected'
        self.updated = True
        x, y = center(self.box)
        self.smoother = Smoother(x, y, self.height())
        self.number = number
        self.mean_movement = 0
                
    def track(self, frame):
        if self.tracker is None:
            self.tracker = cv2.TrackerGOTURN_create()
        if self.status != 'smoothed':
            left, top, right, bottom = self.box
            self.tracker.init(frame, (left, top, right-left, bottom-top))
            self.status = 'tracked'
        
    def update(self, frame, horses):
        logger.info(f'### update horse {self.number} {self.status}')
        try_tracking = False
        if self.detected():
            self.status = 'detected'
            logger.info(f'detected horse {self.number}')
        elif self.status == 'smoothed':
            intersects = False
            for horse in horses:
                if horse == self: continue
                if horse.intersect_fixed(self.box) and not horse.updated and horse.status != 'smoothed':
                    intersects = True
                    logger.info(f'smoothed horse {self.number} intersects with horse {horse.number}')
                logger.info([f'smooth {self.number}:', horse.intersect_fixed(self.box), horse.updated, horse.status != 'smoothed', intersects])
            if intersects:
                logger.info(f'smoothed horse {self.number}')
                logger.info(f'movement of horse {self.number}: {self.movement_history}')
                self.status = 'smoothed'
                left, top, right, bottom = self.box
                left += self.mean_movement; right += self.mean_movement
                logger.info(f'add {self.mean_movement} onto left and right of horse {self.number}')
                box = left, top, right, bottom
                x, y = center(box)
                box = fixed_box(x, y, self.height())
                self.update_direction(box)
                self.box = box
                left, top, right, bottom = box
                self.tracker.init(frame, (left, top, right-left, bottom-top))
            else:
                logger.info(f'horse {self.number} will try to track')
                left, top, right, bottom = self.box
                self.tracker.init(frame, (left, top, right-left, bottom-top))
                try_tracking = True
        if self.status == 'tracked' or try_tracking:
            _, box = self.tracker.update(frame)
            left, top = (int(box[0]), int(box[1]))
            right, bottom = (int(box[0] + box[2]), int(box[1] + box[3]))
            box = left, top, right, bottom
            intersects = False
            for horse in horses:
                if horse == self: continue
                logger.info(f'trytracking: {try_tracking}')
                if try_tracking:
                    if horse.intersect(box) and not horse.updated and horse.status != 'smoothed':
                        logger.info(f'tracker of horse {self.number} intersects with horse {horse.number}')
                        intersects = True
                    logger.info([f'trytrack {self.number}:', horse.intersect(self.box), not horse.updated, horse.status != 'smoothed', intersects])
                else:
                    if horse.intersect_fixed(box) and not horse.updated and horse.status != 'smoothed':
                        logger.info(f'tracker of horse {self.number} intersects_fixed with horse {horse.number}')
                        intersects = True
                    logger.info([f'trytrack {self.number}:', horse.intersect_fixed(self.box), not horse.updated, horse.status != 'smoothed', intersects])
            if intersects:
                logger.info('intersects')
                self.status = 'smoothed'
            else:
                logger.info(f'tracked horse {self.number}')
                self.status = 'tracked'
                self.box = box
        x, y = center(self.box)
        x, y, height = self.smoother.update(x, y, self.height())
        self.smooth_box = fixed_box(x, y, height)
        self.updated = True
                
    def tracking(self):
        return self.tracker is not None
        
    def update_direction(self, box):
        new_x, _ = center(box)
        old_x, _ = center(self.box)
        movement = new_x - old_x
        self.movement_history[:-1] = self.movement_history[1:]
        self.movement_history[-1] = movement
        
    def direction(self):
        if self.mean_direction() > 0:
            return 1
        else:
            return -1
            
    def offset(self, box):
        horse_x, _ = center(self.box)
        box_x, _ = center(box)
        if box_x - horse_x > 0:
            return 1
        else:
            return -1
            
    def mean_direction(self):
        return np.mean(self.movement_history)
        
    def detect(self, box):
        self.update_direction(box)
        self.mean_movement = self.mean_direction()
        self.status = 'detected'
        self.last_detected = 0
        self.tracker = None
        self.box = box
        
    def detected(self):
        return self.tracker is None
        
    def intersect(self, box):
        l1, t1, r1, b1 = self.box
        l2, t2, r2, b2 = box
        if l1 > r2 or l2 > r1:
            return False
        if t1 > b2 or t2 > b1:
            return False
        return True
        
    def intersect_fixed(self, box):
        l1, t1, r1, b1 = self.smooth_box
        l2, t2, r2, b2 = box
        if l1 > r2 or l2 > r1:
            return False
        if t1 > b2 or t2 > b1:
            return False
        return True
        
    def smooth_center(self):
        return center(self.smooth_box)

    def height(self):
        _, top, _, bottom = self.box
        return bottom - top
        
    def distance(self, box):
        p1 = center(self.box)
        p2 = center(box)
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance
        
    def allowed_distance(self):
        _, top, _, bottom = self.smooth_box
        factor = (bottom - top) * 0.2
        return self.last_detected * factor + factor
        
    def gone(self):
        return self.last_detected >= 100
        
    def draw(self, frame):
        if self.status == 'detected':
            color = red
        elif self.status == 'tracked':
            color = blue
        elif self.status == 'smoothed':
            color = yellow
        left, top, right, bottom = self.smooth_box
        x, y = center(self.smooth_box)
        cv2.rectangle(frame, (left, top), (right, bottom), green, 5)
        cv2.putText(frame, str(self.number), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 2, color)
        left, top, right, bottom = self.box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 5)
        
    def draw_smooth(self, frame):
        left, top, right, bottom = self.smooth_box
        x, y = center(self.smooth_box)
        cv2.rectangle(frame, (left, top), (right, bottom), green, 5)