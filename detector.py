from PIL import Image
import cv2
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from box_detector import BoxDetector


class Detector():
    def __init__(self,draw=False):
        self.d = BoxDetector()

        self.draw=draw


    def detect(self,frame): # return
        image = Image.fromarray(frame)
        boxes, scores = self.d.detect_boxes(image)

        if len(boxes) == 0:
            return False, None, frame

        if self.draw==True:
            for box in boxes:
                left, top, right, bottom = box
                if self.draw:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 1)

        return True, boxes, frame





