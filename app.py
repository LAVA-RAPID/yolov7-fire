from __future__ import print_function

import argparse
import cv2
import torch
import torch.nn as nn
import logging
import json
import os

from models.experimental import Ensemble
from models.common import Conv
from utils.general import non_max_suppression

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_TEMPLATE = "env.detection"

class YOLOv7_Main():
    def __init__(self, args, weightfile):
        logging.info("Initializing YOLOv7_Main...")
        try:
            logging.info(f"Checking device availability...")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")

            logging.info("Creating Ensemble model...")
            self.model = Ensemble()

            logging.info(f"Loading weight file: {weightfile}")
            ckpt = torch.load(weightfile, map_location=self.device)
            logging.info("Weight file loaded successfully")

            logging.info("Appending model to Ensemble...")
            self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
            logging.info("Model appended successfully")

            logging.info("Performing compatibility updates...")
            for m in self.model.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True  # pytorch 1.7.0 compatibility
                elif type(m) is nn.Upsample:
                    m.recompute_scale_factor = None  # torch 1.11.0 compatibility
                elif type(m) is Conv:
                    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            logging.info("Compatibility updates completed")

            logging.info("Converting model to half precision...")
            self.model = self.model.half()
            logging.info("Model converted to half precision")

            logging.info("Setting model to eval mode...")
            self.model.eval()
            logging.info("Model set to eval mode")

            logging.info("YOLOv7_Main initialization completed successfully")
        except Exception as e:
            logging.info(f"Error during YOLOv7_Main initialization: {str(e)}")
            raise

    def pre_processing(self, frame):
        logging.info("Starting pre-processing of image")
        try:
            logging.debug("Resizing image to 640x640")
            sized = cv2.resize(frame, (640, 640))
            
            logging.debug("Normalizing image")
            image = sized / 255.0
            
            logging.debug("Transposing image dimensions")
            image = image.transpose((2, 0, 1))
            
            logging.debug("Converting image to PyTorch tensor")
            image = torch.from_numpy(image).to(self.device).half()
            
            logging.debug("Adding batch dimension to tensor")
            image = image.unsqueeze(0)
            
            logging.info("Pre-processing completed successfully")
            return image
        except Exception as e:
            logging.info(f"Error during pre-processing: {str(e)}")
            raise

    def inference(self, image):
        logging.info("Starting inference")
        try:
            logging.debug("Running model inference")
            with torch.no_grad():
                pred = self.model(image)[0]
            
            logging.info("Inference completed successfully")
            return pred
        except Exception as e:
            logging.info(f"Error during inference: {str(e)}")
            raise
          
def list_directories_and_contents(path='.'):
    for root, dirs, files in os.walk(path):
        logging.info(f"Directory: {root}")
        if dirs:
            logging.info("Subdirectories:")
            for dir in dirs:
                logging.info(f"  {dir}")
        if files:
            logging.info("Files:")
            for file in files:
                logging.info(f"  {file}")

def run(args):
    with Plugin() as plugin, Camera(args.stream) as camera:
        classes = {0: 'fire', 1: 'smoke'}
        logging.info(f'Target objects: fire, smoke')
        
        logging.info("Listing directories and contents...")
        list_directories_and_contents()
        
        
        try:
            yolov7_main = YOLOv7_Main(args, args.weight)
            logging.info("YOLOv7_Main object initialized successfully")
        except Exception as e:
            logging.info(f"Error initializing YOLOv7_Main: {str(e)}")
            return

        logging.info(f'Model {args.weight} loaded')
        plugin.publish("env.model.loaded", f"{args.weight} loaded")
        logging.info(f'Confidence threshold is set to {args.conf_thres}')
        logging.info(f'IOU threshold is set to {args.iou_thres}')
        
        sampling_countdown = args.sampling_interval
        if args.sampling_interval >= 0:
            logging.info(f'Sampling enabled -- occurs every {args.sampling_interval}th inferencing')

        logging.info("Fire and smoke detection starts...")
        for sample in camera.stream():
            do_sampling = sampling_countdown == 0
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                sampling_countdown = args.sampling_interval

            frame = sample.data
            image = yolov7_main.pre_processing(frame)
            pred = yolov7_main.inference(image)
            
            # Publish raw YOLO output
            raw_output = pred[0].cpu().numpy().tolist()  # Convert to list for JSON serialization
            plugin.publish('env.yolo.raw_output', json.dumps(raw_output), timestamp=sample.timestamp)
            logging.info("Published raw YOLO output")

            results = non_max_suppression(
                pred,
                args.conf_thres,
                args.iou_thres,
                agnostic=True)[0]

            found = {'fire': 0, 'smoke': 0}
            w, h = frame.shape[1], frame.shape[0]
            for x1, y1, x2, y2, conf, cls in results:
                object_label = classes[int(cls)]
                l, t, r, b = x1 * w/640, y1 * h/640, x2 * w/640, y2 * h/640
                rounded_conf = int(conf * 100)
                if do_sampling:
                    frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255,0,0), 2)
                    frame = cv2.putText(frame, f'{object_label}:{rounded_conf}%', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                found[object_label] += 1

            detection_stats = 'Detected: ' + ' '.join(f'{obj} [{count}]' for obj, count in found.items())
            logging.info(detection_stats)

            for object_found, count in found.items():
                plugin.publish(f'{TOPIC_TEMPLATE}.{object_found}', count, timestamp=sample.timestamp)

            if do_sampling:
                sample.data = frame
                sample.save('sample.jpg')
                plugin.upload_file('sample.jpg', timestamp=sample.timestamp)
                logging.info("Uploaded sample")

            if not args.continuous:
                break

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO v7 Fire and Smoke Detection')
    parser.add_argument('-weight', type=str, default='yolov7-fire.pt', help='model.pt path(s)')
    parser.add_argument('-stream', type=str, default="bottom_camera", help='ID or name of a stream, e.g. sample')
    parser.add_argument('-conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-continuous', action='store_true', default=True, help='Flag to run this plugin forever')
    parser.add_argument('-sampling-interval', type=int, default=-1, help='Sampling interval between inferencing')
    parser.add_argument('-debug', action='store_true', default=True, help='Debug flag')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    run(args)