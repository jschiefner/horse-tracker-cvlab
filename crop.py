import cv2
import numpy as np

frame_width = 3840
frame_height = 2160
#base=frame_height*0.05 # nutze base wenn der zoom auf distanzierte Reiter zu  "aggressiv" ist
# dann wird gewahrleistet das die box eine ungefahre mindestgrose haben
base=0
ratio = frame_width / frame_height

class Cropper():
    def __init__(self,ratio=ratio, zoom=2,base=base):
        self.ratio = ratio
        self.zoom = zoom
        self.base = base # gets always added to cropbox
        
    def crop(self, box, frame):
        left, top, right, bottom = box
        x = np.mean((left, right)).astype(int)
        y = np.mean((top, bottom)).astype(int)
        h = bottom - top
        return self._crop_with_center(frame, x, y, h)
        
    def _crop_with_center(self, frame, x, y, h):
        max_h, max_w, _ = frame.shape

        crop_h = int(h*self.zoom + self.base)
        crop_w = int(crop_h*self.ratio)


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
        if False: # set true for debug reasons
            print("lrtb", left, right, top, bottom)
            print("cutout size:",cutout.shape[0:2])
            print("crop_h_w",crop_h,crop_w)
        assert (cutout.shape[0:2] == (crop_h, crop_w))
        resized = cv2.resize(cutout, (frame_width, frame_height))
        return resized

