# %% imports

import cv2
import matplotlib.pyplot as plt
import numpy as np
def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14.4, 25.6))
    plt.imshow(img)
images = []
for i in range(6, 10):
    images.append(cv2.imread(f'data/images/GOPR8291/0022{i}.png'))
import random as rng
rng.seed(12345)
height = 2160
width = 3840
# %% action

img = images[1]
show(img)
median = np.median(images, axis=0).astype(np.uint8)
show(median)
diff = cv2.subtract(median, img)
show(diff)

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
show(gray)
blurred = cv2.GaussianBlur(gray, (51,51), 50)
show(blurred)
_, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
show(thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
dilated = cv2.dilate(thresh, kernel)
show(dilated)
eroded = cv2.erode(dilated, kernel)
show(eroded)
# [contours], hierarchy = cv2.findContours(eroded.copy(), cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_EXTERNAL)
contours, hierarchy = cv2.findContours(eroded, 1, 2) # evtl ohne dilate und erode
cnt = contours[0] # attention here: index could be out of bounds == no contour found!
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
show(img)
