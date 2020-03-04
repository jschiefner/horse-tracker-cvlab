import cv2

class Writer():
    def __init__(self,filename,h,w):
        self.h=h
        self.w=w
        if filename[-4:]==".mp4":
            filename=filename[:-4]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename+".mp4", fourcc, 25, (w,h))

    def write(self,frame):
        outframe = cv2.resize(frame,(self.w,self.h))
        self.out.write(outframe)

    def save(self):
        print("Saved")
        self.out.release()

    def __del__(self):
        self.out.release()
