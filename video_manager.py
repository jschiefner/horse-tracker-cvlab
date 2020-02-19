import cv2
import progress
from progress.bar import Bar
from progress.spinner import Spinner
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('horse')

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height
plot_width = 14

def show_frame(frame):
    plt.close()
    plt.figure(figsize=(plot_width, plot_width * ratio))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

class VideoManager():
    def __init__(self, input, output, max_frames, skip, show=True):
        self.cap = cv2.VideoCapture(input)
        if not self.cap.isOpened():
            logger.info('Unable to read video feed')
            exit(1)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter(output, fourcc, 25, (frame_width, frame_height))
        self.count = 0
        self.max_frames = max_frames
        if skip > 0: self.skip(skip)
        self.bar = Bar('Processing frames', max=max_frames)
        self.output = output
        self.show = show
        
    def skip(self, frames):
        spinner = Spinner(f'Skipping {frames} Frames... ')
        frame_number = 0
        while frame_number < frames:
            spinner.next()
            ret, _ = self.cap.read()
            if ret: frame_number += 1
        logger.info(f'Skipped {frames} frames')
    
    def getWidth(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def getHeight(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def getFPS(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def read(self):
        self.bar.next()
        logger.info('')
        logger.info(f'{self.count+1}/{self.max_frames}')
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.count += 1
                return frame
    
    def write(self, frame):
        if self.show: show_frame(frame)
        self.out.write(frame)
        
    def close(self):
        self.bar.finish()
        logger.info(f'Processed \033[92m{self.count}\033[00m/{self.max_frames} frames.')
        logger.info(f'Saving result to \033[92m{self.output}\033[00m.')
        logger.info('')
        logger.info('################################################################################')
        logger.info('')
        self.cap.release()
        self.out.release()