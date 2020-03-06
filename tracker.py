# %% imports

import cv2
import tensorflow as tf
from math import sqrt

# %% action
model = tf.keras.models.load_model('model_data/model-best(1).h5')

old = cv2.imread('data/goturn/Luedinghausen/7_target/000001.jpg')
new = cv2.imread('data/goturn/Luedinghausen/7_searching/000001_5.jpg')

model.predict([[old], [new]])

left, right, top, bottom = prediction[0]
model_path = 'model_data/...' # TODO: anpassen
out_res = 227
frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height
sqrt2 = sqrt(2)

# frame: new frame,
# box: current box from horse (to crop image)

def scale(frame, box):
    left, top, right, bottom = box
    return cv2.resize(frame[top:bottom, left:right], (out_res, out_res))
    
def calc_cropped_box(box):
    left, top, right, bottom = box
    x = np.mean((left, right)).astype(int)
    y = np.mean((top, bottom)).astype(int)
    height = bottom-top
    offset = round(sqrt2*height/2)
    left, top, right, bottom = x-offset, y-offset, x+offset, y+offset
    return left, top, right, bottom

class Tracker():
    def __init__(self, frame, box):
        self.old_frame = scale(frame, box)
    
    def predict(frame, box):
        print(box)
        cropped_box = calc_cropped_box(box)
        old = self.old_frame
        new = scale(frame, cropped_box)
        fleft, ftop, _, _ = cropped_box
        left, top, right, bottom = model.predict([[old], [new]])[0]
        left, top, right, bottom = round(left*out_res), round(top*out_res), round(right*out_res), round(bottom*out_res)
        left, top, right, bottom = flet+left, ftop+top, fleft+right, ftop + bottom
        print(left, top, right, bottom)
        self.old_frame = new
        return left, top, right, bottom
        