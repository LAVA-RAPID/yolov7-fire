from __future__ import print_function

import numpy as np
import time
import argparse
import cv2
import datetime
import os
import torch
import torch.nn as nn
import logging

from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_TEMPLATE = "env.count"

def load_class_names(namesfile):
    class_names = {}
    with open(namesfile, 'r') as fp:
        for index, class_name in enumerate(fp):
            class_names[index] = class_name.strip()
    return class_names

class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.model = Ensemble()
        ckpt = torch.load(weightfile, map_location=self.device)
        self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.model = self.model.half()
        self.model.eval()

    def pre_processing(self, frame):
        sized = cv2.resize(frame, (640, 640))
        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        return image.unsqueeze(0)

    def inference(self, image):
        with torch.no_grad():
            return self.model(image)[0]

def run(args):
    with Plugin() as plugin, Camera(args.stream) as camera:
        classes_dict = load_class_names("coco.names")
        target_objects = list(classes_dict.values()) if args.all_objects else (args.object or list(classes_dict.values()))
        classes = [index for index, target in classes_dict.items() if target in target_objects]
        
        logging.info(f'target objects: {" ".join(target_objects)}')
        logging.debug(f'class numbers for target objects are {classes}')
        
        yolov7_main = YOLOv7_Main(args, args.weight)
        logging.info(f'model {args.weight} loaded')
        logging.info(f'cut-out confidence level is set to {args.conf_thres}')
        logging.info(f'IOU level is set to {args.iou_thres}')
        
        sampling_countdown = args.sampling_interval
        if args.sampling_interval >= 0:
            logging.info(f'sampling enabled -- occurs every {args.sampling_interval}th inferencing')

        logging.info("object counter starts...")
        for sample in camera.stream():
            do_sampling = sampling_countdown == 0
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                sampling_countdown = args.sampling_interval

            frame = sample.data
            image = yolov7_main.pre_processing(frame)
            pred = yolov7_main.inference(image)
            results = non_max_suppression(
                pred,
                args.conf_thres,
                args.iou_thres,
                classes,
                agnostic=True)[0]

            found = {}
            w, h = frame.shape[1], frame.shape[0]
            for x1, y1, x2, y2, conf, cls in results:
                object_label = classes_dict[int(cls)]
                if object_label in target_objects:
                    l, t, r, b = x1 * w/640, y1 * h/640, x2 * w/640, y2 * h/640
                    rounded_conf = int(conf * 100)
                    if do_sampling:
                        frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
                        frame = cv2.putText(frame, f'{object_label}:{rounded_conf}%', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    found[object_label] = found.get(object_label, 0) + 1

            detection_stats = 'found objects: ' + ' '.join(f'{obj} [{count}]' for obj, count in found.items())
            logging.info(detection_stats)

            for object_found, count in found.items():
                plugin.publish(f'{TOPIC_TEMPLATE}.{object_found}', count, timestamp=sample.timestamp)

            if do_sampling:
                sample.data = frame
                sample.save('sample.jpg')
                plugin.upload_file('sample.jpg', timestamp=sample.timestamp)
                logging.info("uploaded sample")

            if not args.continuous:
                break

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO v7 Detection')
    parser.add_argument('-weight', type=str, default='yolov7-fire.pt', help='model.pt path(s)')
    parser.add_argument('-stream', type=str, default="camera", help='ID or name of a stream, e.g. sample')
    parser.add_argument('-object', action='append', help='Object name to count')
    parser.add_argument('-all-objects', action='store_true', default=False, help='Consider all registered objects to detect')
    parser.add_argument('-conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-continuous', action='store_true', default=False, help='Flag to run this plugin forever')
    parser.add_argument('-sampling-interval', type=int, default=-1, help='Sampling interval between inferencing')
    parser.add_argument('-debug', action='store_true', default=False, help='Debug flag')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    run(args)