# %% imports

import os
import cv2
import numpy as np

# %% action

label_file_path = 'goturn_label' # TODO: anpassen
with open(label_file_path, 'r') as file:
    lines = file.readlines()

length = len(lines)
target_array = np.empty((length, 227, 227, 3), dtype=np.uint8)
searching_array = np.empty((length, 227, 227, 3), dtype=np.uint8)
coord_array = np.empty((length, 4), dtype=np.float64)

for index, line in enumerate(lines):
    target, searching, left, top, right, bottom = line.split(',')
    left, top, right, bottom = np.float64(left), np.float64(top), np.float64(right), np.float64(bottom)
    target = cv2.imread('bla')
    searching = cv2.imread(searching)
    target_array[index] = target
    searching_array[index] = searching
    coord_array[index] = [left, top, right, bottom]

# use: target_array, searching_array, coord_array