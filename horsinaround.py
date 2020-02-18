import sys
import cv2
from crop import Cropper
from detector import Detector
from reader import Reader
from smooth import Smoother
from writer import Writer
from progress.bar import Bar

if len(sys.argv)<=1:
    print("Usage: horsinaround.py inputfile outfile skipframes=0 framestowrite modvalue")
    exit(0)

filename = sys.argv[1]
skip = int(sys.argv[3])
framestowrite = int(sys.argv[4])

show = False

modvalue = int(sys.argv[5])



reader = Reader(filename=filename)
reader.skipFrames(skip)
print("fps:",reader.getFPS())
height = reader.getHeight()
framewidth = reader.getWidth()
midx = int(reader.getWidth()/2.)
midy = int(height/2.)

writer = Writer(sys.argv[2],480,720)
cropper = Cropper()
detector = Detector(draw=show)
smoother = Smoother(initx=midx,inity=midy,inith=height)



bar = Bar('Processing frames', max=framestowrite)
hhh = []

failedcounter=0

for x in range(framestowrite):
    bar.next()
    frame = reader.read()
    orig_frame = frame.copy()

    if failedcounter>0 or x%modvalue==0:
        if failedcounter<3 or failedcounter%10==0:
            success, box, frame = detector.detect(frame)
            if success:
                failedcounter = 0
                # only 1 horse at a time

                left, top, right, bottom = box


                height = bottom - top
                midx = left + (right - left) / 2
                midy = top + height / 2
                if height < 0: height *= -1
                hhh.append(height)
                if midx < 0: midx *= -1
                if midy < 0: midy *= -1
                midx, midy, height = smoother.update(midx, midy, height)

                # tracker
                # TODO init tracker
                initBB = (left, top, right - left, bottom - top)
                #print(initBB)
                tracker = cv2.TrackerCSRT_create()
                #tracker = None
                #tracker = cv2.TrackerGOTURN_create()
                tracker.init(orig_frame, initBB)
            else:
                failedcounter+=1
        else: #failcounter >=3 oder nicht %10==0
            failedcounter+=1
    else:
        success, box = tracker.update(orig_frame)
        #print(box)
        if success:
            failedcounter=0
            # only 1 horse at a time

            left, top, width, height = box
            if show==True:
                cv2.rectangle(frame, (int(left), int(top)), (int(left+width), int(top+height)), (0, 255, 0), 1)

            midx = left + (width) / 2
            midy = top + height / 2
            if height < 0: height *= -1
            hhh.append(height)
            if midx < 0: midx *= -1
            if midy < 0: midy *= -1
            midx, midy, height = smoother.update(midx, midy, height)
        else:
            failedcounter+=1




    cutout = cropper.crop(orig_frame,midx,midy,height)

    if show: cv2.imshow("frame", frame)
    if show: cv2.imshow("cutout",cutout)
    writer.write(cutout)

    if show:
        key= cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            writer.save()
            break
        elif key==ord("s"):
            reader.skipFrames(reader.getFPS())

cv2.destroyAllWindows()
print("")
writer.save()
print("Done!")

print(hhh)