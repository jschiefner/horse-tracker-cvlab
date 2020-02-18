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
# %% action

img = images[1]
median = np.median(images, axis=0).astype(np.uint8)
diff = cv2.subtract(median, img)
show(diff)

# %%

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (51,51), 50)
show(gray)
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
show(thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
dilated = cv2.dilate(thresh, kernel)
eroded = cv2.erode(dilated, kernel)
show(eroded)
[contours], hierarchy = cv2.findContours(eroded, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_EXTERNAL)
# contours_poly = [None] * len(contours)
# boundRect = [None] * len(contours)
# centers = [None]*len(contours)
# radius = [None]*len(contours)
# for i, c in enumerate(contours):
#     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])
#     centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
# boundRect
# drawing = np.zeros((eroded.shape[0], eroded.shape[1], 3), dtype=np.uint8)
# for i in range(len(contours)):
#     color = (255,255,255)
#     cv2.drawContours(drawing, contours_poly, i, color)
#     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#       (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
#     cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
# show(drawing)
