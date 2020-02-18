from filterpy.kalman import KalmanFilter
from filterpy.kalman import FixedLagSmoother
from filterpy.common import Q_discrete_white_noise
import numpy as np


class Smoother():
    def __init__(self, initx=0., inity=0.,inith=0):
        inith = float(inith)
        # kalman stuff
        self.xfilter = KalmanFilter(dim_x=2, dim_z=1)
        # todo initialize to better values
        self.xfilter.x = np.array([[initx], [0.]])
        self.xfilter.F = np.array([[1., 1.], [0., 1.]])
        self.xfilter.H = np.array([[1., 0.1]])
        self.xfilter.P *= 10 ** 4
        self.xfilter.R = 50.0
        self.xfilter.Q = Q_discrete_white_noise(2, 1.0, 1.0)

        self.yfilter = KalmanFilter(dim_x=2, dim_z=1)
        # todo initialize to better values
        self.yfilter.x = np.array([[inity], [0.]])
        self.yfilter.F = np.array([[1., 1.], [0., 1.]])
        self.yfilter.H = np.array([[1., 0.0]])
        self.yfilter.P *= 10.0 ** 4
        self.yfilter.R = 50.0
        self.yfilter.Q = Q_discrete_white_noise(2, 1.0, 1.0)

        self.hfilter = KalmanFilter(dim_x=2, dim_z=1)
        # todo initialize to better values
        self.hfilter.x = np.array([[inith], [0.]])
        self.hfilter.F = np.array([[1., 1.], [0., 0.]])
        self.hfilter.H = np.array([[1., 0.5]])
        self.hfilter.P *= 10.0 ** 4
        self.hfilter.R = 50.0
        self.hfilter.Q = Q_discrete_white_noise(2, 1.0, 1.0)

        self.fls = FixedLagSmoother(dim_x=2, dim_z=1, N=10)

        self.fls.x = np.array([[inith],[.5]])
        self.fls.F = np.array([[1., 1.],[0., 1.]])
        self.fls.H = np.array([[1., 0.]])

        self.fls.P *= 10.0**4
        self.fls.R *= 100.0
        self.fls.Q *= 0.001

    def update(self, x, y, h):
        self.xfilter.predict()
        self.xfilter.update(x)
        self.yfilter.predict()
        self.yfilter.update(y)
        #self.hfilter.predict()
        #self.hfilter.update(h)
        #self.fls.predict()
        self.fls.smooth(h)
        print("xsmooth",self.fls.xSmooth)
        print(self.fls.xSmooth[-1])

        return self.xfilter.x[0], self.yfilter.x[0], self.fls.xSmooth[-1][0][0]


