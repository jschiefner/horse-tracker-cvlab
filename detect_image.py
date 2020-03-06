# imports

import cv2
import background_box_detector
import yolo_box_detector
yolo = yolo_box_detector.BoxDetector(model_path='logs/000/super_intermediate2.h5', anchors_path='model_data/yolo_anchors.txt')
tiny = yolo_box_detector.BoxDetector(model_path='logs/000/tiny-yolo-intermediate1.h5', anchors_path='model_data/tiny_yolo_anchors.txt')
red = (0,0,255)
scale_factor = 0.2
frame_width = round(3840 * scale_factor)
frame_height = round(2160 * scale_factor)

def scale(frame):
    return cv2.resize(frame, (frame_width, frame_height))

def draw(box, frame):
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), red, 10)

out_folder = 'out/abgabe'
while True:
    print('Input,Output:', end=' ')
    try:
        inpath, outpath = input().split(',')
        inpath, outpath = 'data/images/'+inpath, 'out/abgabe/'+outpath
        print(f'input: {inpath}, output: {outpath}')
        img = cv2.imread(inpath)
        if img is None:
            print('input parameter does not exist')
            continue
        yolo_img = img.copy()
        tiny_img = img.copy()
        boxes, _ = yolo.detect_boxes(img)
        for box in boxes: draw(box, yolo_img)
        boxes, _ = tiny.detect_boxes(img)
        for box in boxes: draw(box, tiny_img)
        cv2.imwrite(outpath + '_yolo.png', scale(yolo_img))
        cv2.imwrite(outpath + '_tiny.png', scale(tiny_img))
    except Exception as e:
        print(e)
        continue
