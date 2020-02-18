

class Tracker():
    def __init__(self):
        self.tracker = None
        pass

    def update(self,frame):
        return self.tracker.update(frame)

    def reinit(self,box):
        self.tracker=None
        self.tracker=None # init to newtracker

    def __del__(self):
        #self.tracker
        pass