import cv2

class Reader():
    def __init__(self,filename=0,show=False):
        self.cap = cv2.VideoCapture(filename)

        if self.cap.isOpened():
            print("File loaded")
        else:
            print("File not loaded")
            raise Exception("shisthiatasfa")
        self.show = show

    def getWidth(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def getHeight(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def read(self,newsize=(0,0)):
        if not self.cap.isOpened():
            print("File not loaded")
            return None

        ret, frame = self.cap.read()
        while not ret:
            ret, frame = self.cap.read()

        if newsize!=(0,0):
            frame = cv2.resize(frame,newsize)
        if self.show==True:
            cv2.imshow("frame", frame) # eventuell anders benennen
        return frame

    def skipFrames(self,frames_to_skip):
        try:
            self.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.cap.get(cv2.CAP_PROP_POS_FRAMES)+frames_to_skip)
            print("Skipped " + str(frames_to_skip) + " frames")
        except Exception as e:
            print("Frame skippen fehlgeschlagen")
            print(e)

    def getFPS(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def __del__(self):
        self.cap.release()