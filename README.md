# Lab course Horse detection

 - Repository based on https://github.com/qqwweee/keras-yolo3, to train follow the instructions there
 - Copy Folders marked in gitignore to run the examples

### Usage

run horsinaround.py infile outfile skipframes takeframes 

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

#### smoother
 - takes x,y,h and gives back smoothed versions of them

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
