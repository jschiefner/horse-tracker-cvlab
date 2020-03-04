# %% imports

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger('horse')

# %% class

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200,200))
frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height
plot_width = 14
len_images = 3
range_frames = 50
count_before_ready = len_images * range_frames

def show_frame(frame):
    plt.close()
    plt.figure(figsize=(plot_width, plot_width * ratio))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

class BoxDetector():
    def __init__(self):
        self.images = np.empty((len_images, frame_height, frame_width, 3), dtype=int) # change to np.uint8
        self.ready = False
        self.count = 0
        
    def _count(self):
        self.count += 1
        if self.count >= count_before_ready: self.ready = True
        
    def _add_image(self, frame):
        if self.count % range_frames == 0:
            self.images[:-1] = self.images[1:]
            self.images[-1] = frame
        
    def detect_boxes(self, frame):
        self._add_image(frame)
        self._count()
        if not self.ready: return [],[]
        
        logger.info('median')
        median = np.median(self.images, axis=0).astype(np.uint8)
        logger.info('diff')
        diff = cv2.subtract(median, frame)
        logger.info('gray')
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        logger.info('blur')
        blurred = cv2.GaussianBlur(gray, (51,51), 50)
        logger.info('thresh')
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        logger.info('dilate')
        dilated = cv2.dilate(thresh, kernel)
        logger.info('erode')
        eroded = cv2.erode(dilated, kernel)
        contours, hierarchy = cv2.findContours(eroded, 1, 2)
        scores = np.full(len(contours),1) # score is always 1
        boxes = np.empty((len(contours), 4), dtype=int)
        for index, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            left, top, right, bottom = x, y, x+w, y+h
            boxes[index] = (left, top, right, bottom)
        logger.info(f'scores: {scores}, boxes: {boxes}')
        return boxes, scores