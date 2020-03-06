# Lab course Horse detection

 - Repository based on https://github.com/qqwweee/keras-yolo3, to train follow the instructions there
 - Copy Folders marked in gitignore to run the examples

### Usage

Run `python horsinaround.py` with the following arguments:

```
usage: horsinaround.py [-h] [--skip [SKIP]] [--single] [--multiple]
                       [--mode [MODE]]
                       input output frames
```
 - `skip` should be the amount of frames you want to skip at the beginning at the video (defaults to 0).
 - `--single` should be passed without parameter if a single horse should be tracked
 - `--multiple` should be passed without parameter if multiple horses should be tracked
 - `--mode` is the horse detection mode. Can be one of `tinyyolo`, `yolo` and `background`
 - `input` is the input video file path
 - `output` is the path for the video output. The `.avi` extension will automatically be appended, so only the filename needs to be given here
 - `frames` is the amount of frames that should be processed. 0 to go to end of file


### Dependencies

 - Global
   - libopencv-dev
   - python-opencv
 - Python (3.6.9)
   - tensorflow (1.6.0)
   - keras (2.1.5)
   - numpy
   - imutils
   - opencv-python
   - Pillow
   - matplotlib
   - h5py
   - cvutils
   - opencv-contrib-python (4.0.0.21)
   - progress
   - filterpy
   
### Aufbau

#### cropper
- takes a frame and gives back a cropped out part in the correct aspect ratio
- 

#### smoother
 - takes x,y,h and gives back smoothed versions of them
 - uses the last 50 frames

#### reader
 - read
 - skipframes

#### detector
 - uses box detector
 - gives back box

#### tracker

#### writer

#### horsinaround
 - main
 - argparse
