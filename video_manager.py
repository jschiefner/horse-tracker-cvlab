import cv2
import progress
from progress.bar import Bar
from progress.spinner import Spinner
import matplotlib.pyplot as plt

frame_width = 3840
frame_height = 2160
ratio = frame_width / frame_height
plot_width = 12

def show(frame):
    plt.close()
    plt.figure(figsize=(plot_width, plot_width * ratio))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

class VideoManager():
    def __init__(self, input, output, max_frames, skip):
        self.cap = cv2.VideoCapture(input)
        if not self.cap.isOpened():
            print('Unable to read video feed')
            exit(1)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter(output, fourcc, 25, (frame_width, frame_height))
        self.count = 0
        self.max_frames = max_frames
        if skip > 0: self.skip(skip)
        self.bar = bar = Bar('Processing frames', max=max_frames)
        self.output = output
        
    def skip(self, frames):
        spinner = Spinner(f'Skipping {frames} Frames... ')
        frame_number = 0
        while frame_number < frames:
            spinner.next()
            ret, _ = self.cap.read()
            if ret: frame_number += 1
        print(f'Skipped {frames} frames')
        
    def read(self):
        self.bar.next()
        if self.count % 10 == 0: print('')
        print(f'{self.count+1}/{self.max_frames} ', end='')
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.count += 1
                return frame
    
    def write(self, frame):
        show(frame)
        self.out.write(frame)
        
    def close(self):
        self.bar.finish()
        print(f'Processed \033[92m{self.count}\033[00m/{self.max_frames} frames.')
        print(f'Saving result to \033[92m{self.output}\033[00m.')
        self.cap.release()
        self.out.release()