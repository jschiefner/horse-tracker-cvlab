import sys
import argparse
import warnings
import logging
import time
# import box_detector

logger = logging.getLogger('horse')
logger.addHandler(logging.FileHandler('out/log.txt', 'a'))
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Horsing Around')
parser.add_argument('input', metavar='input', type=str, help='path to the input video')
parser.add_argument('output', metavar='output', type=str, help='path to the vidoe output')
parser.add_argument('frames', metavar='frames', type=str, help='Amount of frames to be processed in total, 0 to go to EOF')
parser.add_argument('--skip', dest='skip', action='store', nargs='?', type=int, default=0, help='Choose amount of frames to be skipped')
parser.add_argument('--single', dest='single', action='store_true', help='Expcting only a single horse')
parser.add_argument('--multiple', dest='multiple', action='store_true', help='Expcting multiple horses in the video')
parser.add_argument('--mode', dest='mode', action='store', nargs='?', type=str, default='yolo', help='Choose the Detection Mode: tinyyolo, yolo or background')
args = parser.parse_args()

if args.single == args.multiple:
    print('either pass --single or --multiple')
    exit(1)
if args.mode != 'tinyyolo' and args.mode != 'yolo' and args.mode != 'background':
    print('mode needs to be either "tinyyolo", "yolo" or "background"')
    exit(1)

input, output, frames, skip, mode = args.input, args.output, int(args.frames), int(args.skip), args.mode

if mode == 'background':
    print('loading Background Subtraction box detector')
    from background_box_detector import BoxDetector
    detector = BoxDetector()
    print('adding 150 frames for background_box_detector to calibrate')
    frames += 150
else:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from yolo_box_detector import BoxDetector
        if mode == 'yolo':
            print('loading YOLO box detector')
            detector = BoxDetector(model_path='logs/000/super_intermediate2.h5', anchors_path='model_data/yolo_anchors.txt')
        else:
            print('loading Tiny-YOLO box detector')
            detector = BoxDetector(model_path='logs/000/tiny-yolo-intermediate1.h5', anchors_path='model_data/tiny_yolo_anchors.txt')
            
if args.single:
    print('loading single horse manager')
    from detect_track_single import Manager
else:
    print('loading multiple horse manager')
    from detect_track_multiple import Manager

manager = Manager(input, output, frames, skip, False, detector)
print("Zeitpunnkt A",time.time())
manager.initialize()
if frames==0:
    frames=int(manager.getFrameCount()-skip)-1
print(frames)
for i in range(frames-1):
    try:
        manager.update()
    except:
        print('Exception!')
        break
manager.video.close()
print("Zeitpunkt B",time.time())