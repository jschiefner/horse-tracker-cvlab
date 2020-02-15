from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np


class Smooth():
    def __init__(self,initx=500,inity=500):
        
        # kalman stuff
        self.xfilter = KalmanFilter(dim_x=2, dim_z=1)
        # todo initialize to better values
        self.xfilter.x = np.array([[initx],[0.]])
        self.xfilter.F = np.array([[1.,1.],[0.,1.]])
        self.xfilter.H = np.array([[1.,0.1]])
        self.xfilter.P *= 10**4
        self.xfilter.R = 50.0
        self.xfilter.Q = Q_discrete_white_noise(2,1.0,1.0)

        
        self.yfilter = KalmanFilter(dim_x=2, dim_z=1)
        # todo initialize to better values
        self.yfilter.x = np.array([[inity],[0.]])
        self.yfilter.F = np.array([[1.,1.],[0.,1.]])
        self.yfilter.H = np.array([[1.,0.1]])
        self.yfilter.P *= 10.0**4
        self.yfilter.R = 50.0
        self.yfilter.Q = Q_discrete_white_noise(2,1.0,1.0)

        # testtest not working sadly
        #self.filter = KalmanFilter(dim_x=4, dim_z=2)
        # todo initialize to better values
        #self.filter.x = np.array([[0.],[0.],[0.],[0.]])
        #self.filter.F = np.array([  [1.,0.,1.,0.],
        #                            [0.,1.,0.,1.],
        #                            [0.,0.,1.,0.],
        #                            [0.,0.,0.,1.]])
        #self.filter.H = np.array([[1.,0.1,1.,0.1]])
        #self.filter.P *= 10.0**4
        #self.filter.R = 50.0
        #self.filter.Q = Q_discrete_white_noise(4,1.0,1.0)
        



        

    def update(self,x,y,h=0):
        self.xfilter.predict()
        self.xfilter.update(x)
        self.yfilter.predict()
        self.yfilter.update(y)
        
        return self.xfilter.x, self.yfilter.x


