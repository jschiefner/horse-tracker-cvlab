from filterpy.kalman import KalmanFilter
from filterpy.kalman import FixedLagSmoother
from filterpy.common import Q_discrete_white_noise
import numpy as np

class Smoother():
    def __init__(self, initx=0., inity=0.,inith=0):
        inith = float(inith)

        # xfilter smoothes the movement on the x axis
        self.xfilter = FixedLagSmoother(dim_x=2, dim_z=1, N=50)
        self.xfilter.x = np.array([[initx], [0.]])
        self.xfilter.F = np.array([[1., 1.], [0., 1.]])
        self.xfilter.H = np.array([[1., 1]])
        self.xfilter.P *= 10 ** 4
        self.xfilter.R = 50.0
        self.xfilter.Q = Q_discrete_white_noise(2, 1.0, 1.0)

        # yfilter smoothes the movement on the y axis
        self.yfilter = FixedLagSmoother(dim_x=2, dim_z=1, N=50)
        self.yfilter.x = np.array([[inity], [0.]])
        self.yfilter.F = np.array([[1., 1.], [0., 1.]])
        self.yfilter.H = np.array([[1., 50.]])
        self.yfilter.P *= 10.0 ** 4
        self.yfilter.R = 50.0
        self.yfilter.Q = Q_discrete_white_noise(2, 1.0, 1.0)

        # hfilter or heightfilter smoothes out the height changes of the boxes
        self.hfilter = FixedLagSmoother(dim_x=2, dim_z=1, N=50)
        self.hfilter.x = np.array([[inith],[.5]])
        self.hfilter.F = np.array([[1., 1.],[0., 1.]])
        self.hfilter.H = np.array([[1., 1.]])
        self.hfilter.P *= 10.0**4
        self.hfilter.R *= 100.0
        self.hfilter.Q *= 0.001
        
    def predict(self, h):
        raise Exception("remove, diese Funktion wird nicht benoetig, lag smooth arbeitet nur mit smooth; nicht mit predict und update")


    def update(self, x, y, h):
        self.xfilter.smooth(x)
        self.yfilter.smooth(y)
        self.hfilter.smooth(h)
        # wenn smooth zu lang kuerzen, benoetigt werden N frames
        if len(self.xfilter.xSmooth) > 500: self.xfilter.xSmooth = self.xfilter.xSmooth[-51:-1]
        if len(self.yfilter.xSmooth) > 500: self.yfilter.xSmooth = self.yfilter.xSmooth[-51:-1]
        if len(self.hfilter.xSmooth) > 500: self.hfilter.xSmooth = self.hfilter.xSmooth[-51:-1]

        return int(round(self.xfilter.xSmooth[-1][0][0])), int(round(self.yfilter.xSmooth[-1][0][0])), int(round(self.hfilter.xSmooth[-1][0][0]))


