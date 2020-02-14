# usage: python detect_and_save_to_file.py <folder path>

# %% argument parse

import sys
if len(sys.argv) != 2:
    print('Pass folder path as first parameter, for example: python detect_and_save_to_file.py data/images/GP028291')
    exit(1)
_, folder_path = sys.argv

# %% imports

import cv2
import numpy as np
from PIL import Image
from glob import glob
import os
import warnings
from progress.bar import Bar
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from box_detector import BoxDetector
    box_detector = BoxDetector()

# %% action

file_list = glob(os.path.join(folder_path, '*.png'))
bar = Bar('Processing Images', max=len(file_list))
for path in file_list:
    bar.next()
    img = cv2.imread(path)
    pil = Image.fromarray(img)
    boxes, scores = box_detector.detect_boxes(pil)
    left, top, right, bottom = boxes[0]
    folder, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)
    out_path = os.path.join(folder, f'{filename}.txt')
    with open(out_path, 'w') as out:
        out.write(f'{left} {top} {right} {bottom}')
bar.finish()