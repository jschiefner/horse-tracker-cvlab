import cv2
import numpy as np

class Cropper():
    def __init__(self,ratio=1.5, zoom=2):
        self.ratio = ratio
        self.zoom = zoom

    def crop(self, frame, x, y, h):
        crop_h = int(h*self.zoom)
        crop_w = int(crop_h*self.ratio)
        # TODO get maximalste groeßte bei vorgegebenen ratio
        max_h, max_w, _ = frame.shape
        if max_h * self.ratio > max_w:
            max_h = int(max_w / self.ratio)
        else:
            max_w = int(max_h * self.ratio)

        if crop_h>max_h:
            crop_h=max_h
        if crop_w>max_w:
            crop_w=max_w

        left = int(x - crop_w / 2)
        right = int(x + crop_w / 2)

        roundingerror = crop_w - right + left  # 640-999+360 = 1
        if roundingerror != 0:
            right += roundingerror

        if left < 0:
            left = 0
            right = crop_w
        if right > frame.shape[1]:
            right = frame.shape[1]
            left = frame.shape[1]  - crop_w

        top = int(y - crop_h / 2)
        bottom = int(y + crop_h / 2)

        roundingerror = crop_h - bottom + top  # 480-699-220 = 1
        if roundingerror != 0:
            bottom += roundingerror

        if top < 0:
            top = 0
            bottom = crop_h
        if bottom > frame.shape[0]:
            bottom = frame.shape[0]
            top = frame.shape[0] - crop_h

        cutout = frame[top:bottom, left:right]
        if False:
            print("lrtb", left, right, top, bottom)
            print("cutout size:",cutout.shape[0:2])
            print("crop_h_w",crop_h,crop_w)
        assert (cutout.shape[0:2] == (crop_h, crop_w))

        return cutout

