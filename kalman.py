import cv2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

class Kalman():
    def __init__(self, box):
        left, top, right, bottom = box
        height = np.mean((top, bottom))
        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.x = np.array([[height],[0.]])
        self.kalman.F = np.array([[1.,1.],[0.,1.]])
        self.kalman.H = np.array([[1.,0.1]])
        self.kalman.P *= 10**4
        self.kalman.R = 50.0
        self.kalman.Q = Q_discrete_white_noise(2,1.0,1.0)
        
    def correct(self, box):
        left, top, right, bottom = box
        height = top - bottom
        self.kalman.update(height)
        
    def predict(self):
        self.kalman.predict()
        prediction = self.kalman.x[0,0]
        return int(prediction)
        
class Kalman2D():
    def __init__(self, box):
        kalman = cv2.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)

        kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)

        kalman.processNoiseCov = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]],np.float32) * 0.03
        self.kalman = kalman
        self.correct(box)

    def correct(self, box):
        left, top, right, bottom = box
        x = np.mean((left, right))
        y = np.mean((top, bottom))
        center = np.array([np.float32(x), np.float32(y)], np.float32)
        self.kalman.correct(center)
        
    def predict(self):
        [x], [y], _, _ = self.kalman.predict()
        return x, y