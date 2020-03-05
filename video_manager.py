import cv2
import progress
from progress.bar import Bar
from progress.spinner import Spinner
import matplotlib.pyplot as plt
import logging
from timeit import default_timer as timer
from crop import Cropper
logger = logging.getLogger('horse')

frame_width = 1080
frame_height = 720
ratio = frame_width / frame_height
plot_width = 14
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
cropper = Cropper(ratio)

def show_frame(frame):
    plt.close()
    plt.figure(figsize=(plot_width, plot_width * ratio))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

class VideoManager():
    def __init__(self, input, output, max_frames, skip, show=False):
        self.cap = cv2.VideoCapture(input)
        if not self.cap.isOpened():
            logger.info('Unable to read video feed')
            exit(1)
        self.out = cv2.VideoWriter(f'{output}.avi', fourcc, 25, (frame_width, frame_height))
        self.smooth_out = cv2.VideoWriter(f'{output}_smooth.avi', fourcc, 25, (frame_width, frame_height))
        self.horses = [None, None, None, None, None]
        self.outs = [None, None, None, None, None]
        self.count = 0
        self.max_frames = max_frames
        if skip > 0: self.skip(skip)
        self.bar = Bar('Processing frames', max=max_frames)
        self.output = output
        self.show = show
        logger.info('################################################################################')
        logger.info(f'Input: \033[92m{input}\033[00m, Output: \033[92m{output}\033[00m')
        logger.info(f'\033[92m{max_frames}\033[00m frames, show while processing: \033[92m{show}\033[00m')
        self.start_time = timer()
        
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
                                
    def horse_out(self, horse):
        index = horse.number
        out = self.outs[index]
        if out is None:
            out = cv2.VideoWriter(f'{self.output}{index}.avi', fourcc, 25, (frame_width, frame_height))
            self.outs[index] = out
        return out
    
    def write(self, raw, frame, smooth, horses):
        if self.show:
            show_frame(frame)
            # show_frame(smooth)
        for horse in horses:
            if horse.number >= 5:
                logger.info('can not support more than 5 horses at once')
                continue
            out = self.horse_out(horse)
            cropped = cropper.crop(horse.smooth_box, raw)
            cropped = cv2.resize(cropped,(frame_width,frame_height))
            # if self.show: show_frame(cropped)
            out.write(cropped)
        self.smooth_out.write(smooth)
        self.out.write(frame)
        
    def close(self):
        self.bar.finish()
        elapsed_seconds = timer()-self.start_time; elapsed_minutes = elapsed_seconds / 60
        logger.info(f'time elapsed: \033[92m{elapsed_seconds}\033[00m seconds => \033[92m{elapsed_minutes}\033[00m minutes.')
        logger.info(f'Processed \033[92m{self.count}\033[00m/{self.max_frames} frames.')
        logger.info(f'Saving result to \033[92m{self.output}\033[00m.')
        logger.info('')
        logger.info('################################################################################')
        logger.info('')
        self.cap.release()
        self.out.release()
        self.smooth_out.release()
        for out in self.outs:
            if out is not None:
                out.release()