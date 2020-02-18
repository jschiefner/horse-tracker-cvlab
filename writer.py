import cv2

class Writer():
    def __init__(self,filename,h,w):
        #self.frameid=1
        self.h=h
        self.w=w
        if filename[-4:]==".mp4":
            filename=filename[:-4]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename+".mp4", fourcc, 25, (w,h))

    def write(self,frame):
        outframe = cv2.resize(frame,(self.w,self.h))
        #cv2.imwrite("frames/"+str(self.frameid)+".jpg", outframe)
        #self.frameid+=1
        self.out.write(outframe)


    def save(self):
        print("Saved")
        self.out.release()


    def __del__(self):
        self.out.release()
