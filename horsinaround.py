# %% imports

import sys
import cv2
from crop import Cropper
from detector import Detector
# from reader import Reader
from smooth import Smoother
# from writer import Writer
from progress.bar import Bar
from video_manager import VideoManager

# %% action

# if len(sys.argv)<=1:
#     print("Usage: horsinaround.py inputfile outfile skipframes=0 framestowrite")
#     exit(0)

# filename = sys.argv[1]
# skip = int(sys.argv[3])
# framestowrite = int(sys.argv[4])

filename = 'data/videos/GP038291.MP4'
outfile = f'out/around{input()}.avi'
skip = 8 * 23
framestowrite = 8 * 23
show = True

video = VideoManager(filename, outfile, framestowrite, skip, show)
print("fps:",video.getFPS())
height = video.getHeight()
width = video.getWidth()
midx = int(width/2.)
midy = int(height/2.)

# writer = Writer(outfile,480,720)
cropper = Cropper()
detector = Detector(draw=show)
smoother = Smoother(initx=midx,inity=midy,inith=height)

bar = Bar('Processing frames', max=framestowrite)
hhh = []

failedlasttime=False
for x in range(framestowrite):
    bar.next()
    frame = video.read()
    orig_frame = frame.copy()
    if failedlasttime or x%6==0:
        success, box, frame = detector.detect(frame)
        if success:
            failedlasttime = False
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
            # tracker = cv2.TrackerGOTURN_create()
            tracker.init(orig_frame, initBB)
        else:
            failedlasttime=True
    else:
        success, box = tracker.update(orig_frame)
        #print(box)
        if success:
            failedlasttime = False
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
            failedlasttime = True

    cutout = cropper.crop(orig_frame,midx,midy,height)
    resized = cv2.resize(cutout, (3840, 2160))
    video.write(resized)

    # if show:
    #     key= cv2.waitKey(1) & 0xFF
    #     if key==ord("q"):
    #         break
    #     elif key==ord("s"):
    #         video.skipFrames(video.getFPS())

# cv2.destroyAllWindows()
print("")
print(hhh)
video.close()

