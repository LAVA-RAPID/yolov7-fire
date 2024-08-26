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
        logging.debug("Initializing YOLOv7_Main...")
        try:
            logging.debug(f"Checking device availability...")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.debug(f"Using device: {self.device}")

            logging.debug("Creating Ensemble model...")
            self.model = Ensemble()

            logging.debug(f"Loading weight file: {weightfile}")
            plugin.publish("env.debug", f"Loading weight file: {weightfile}")
            ckpt = torch.load(weightfile, map_location=self.device)
            logging.debug("Weight file loaded successfully")

            logging.debug("Appending model to Ensemble...")
            self.model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
            logging.debug("Model appended successfully")

            logging.debug("Performing compatibility updates...")
            for m in self.model.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True  # pytorch 1.7.0 compatibility
                elif type(m) is nn.Upsample:
                    m.recompute_scale_factor = None  # torch 1.11.0 compatibility
                elif type(m) is Conv:
                    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            logging.debug("Compatibility updates completed")

            logging.debug("Converting model to half precision...")
            self.model = self.model.half()
            logging.debug("Model converted to half precision")

            logging.debug("Setting model to eval mode...")
            self.model.eval()
            logging.debug("Model set to eval mode")

            logging.debug("YOLOv7_Main initialization completed successfully")
        except Exception as e:
            logging.debug(f"Error during YOLOv7_Main initialization: {str(e)}")
            raise

    def pre_processing(self, frame):
        logging.debug("Starting pre-processing of image")
        try:
            logging.debug("Resizing image to 640x640")
            sized = cv2.resize(frame, (640, 640))
            
            logging.debug("Normalizing image")
            image = sized / 255.0
            
            logging.debug("Transposing image dimensions")
            image = image.transpose((2, 0, 1))
            
            logging.debug("Converting image to PyTorch tensor")
            image = torch.from_numpy(image).to(self.device).half() # converts from fp32 to fp16, doesn't affect performance of model much but lowers memory usage
            
            logging.debug("Adding batch dimension to tensor")
            image = image.unsqueeze(0)
            
            logging.debug("Pre-processing completed successfully")
            return image
        except Exception as e:
            logging.debug(f"Error during pre-processing: {str(e)}")
            raise

    def inference(self, image):
        logging.debug("Starting inference")
        try:
            logging.debug("Running model inference")
            with torch.no_grad():
                pred = self.model(image)[0]
            
            logging.debug("Inference completed successfully")
            return pred
        except Exception as e:
            logging.debug(f"Error during inference: {str(e)}")
            raise
          
def list_directories_and_contents(path='.'):
    for root, dirs, files in os.walk(path):
        logging.debug(f"Directory: {root}")
        if dirs:
            logging.debug("Subdirectories:")
            for dir in dirs:
                logging.debug(f"  {dir}")
        if files:
            logging.debug("Files:")
            for file in files:
                logging.debug(f"  {file}")

def process_frame(frame, yolov7_main, plugin, args, classes, do_sampling=False, timestamp=None):
    image = yolov7_main.pre_processing(frame)
    pred = yolov7_main.inference(image)

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

    plugin.publish(f'{TOPIC_TEMPLATE}', json.dumps(found), timestamp=timestamp)

    if do_sampling or args.testmodel:
        cv2.imwrite('sample.jpg', frame)
        plugin.upload_file('sample.jpg', timestamp=timestamp)
        logging.debug("Uploaded sample")

def run(args):
    with Plugin() as plugin, Camera(args.stream) as camera:
        classes = {0: 'fire', 1: 'smoke'}
        logging.debug(f'Target objects: fire, smoke')

        logging.debug("Listing directories and contents...")
        list_directories_and_contents()

        try:
            yolov7_main = YOLOv7_Main(args, args.weight)
            logging.debug("YOLOv7_Main object initialized successfully")
        except Exception as e:
            logging.debug(f"Error initializing YOLOv7_Main: {str(e)}")
            return

        logging.debug(f'Model {args.weight} loaded')
        plugin.publish("env.model.loaded", f"{args.weight} loaded")
        logging.debug(f'Confidence threshold is set to {args.conf_thres}')
        logging.debug(f'IOU threshold is set to {args.iou_thres}')

        if args.testmodel:
            # Test mode: read a single image
            logging.debug("Running in test mode with a single image")
            frame = cv2.imread('./test/test.jpg')  # Replace with your test image path
            if frame is None:
                logging.error("Failed to load test image")
                return
            process_frame(frame, yolov7_main, plugin, args, classes)
        else:
            sampling_countdown = args.sampling_interval
            if args.sampling_interval >= 0:
                logging.debug(f'Sampling enabled -- occurs every {args.sampling_interval}th inferencing')

            logging.debug("Fire and smoke detection starts...")
            
            for sample in camera.stream():
                do_sampling = False
                if sampling_countdown > 0:
                    sampling_countdown -= 1
                elif sampling_countdown == 0:
                    do_sampling = True
                    sampling_countdown = args.sampling_interval

                frame = sample.data
                process_frame(frame, yolov7_main, plugin, args, classes, do_sampling, sample.timestamp)

                if not args.continuous:
                    break

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO v7 Fire and Smoke Detection')
    parser.add_argument('-weight', type=str, default='yolov7-fire.pt', help='model.pt path(s)')
    parser.add_argument('-stream', type=str, default="bottom_camera", help='ID or name of a stream, e.g. bottom_camera')
    parser.add_argument('-conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-continuous', action='store_true', default=False, help='Flag to run this plugin forever')
    parser.add_argument('-sampling-interval', type=int, default=-1, help='Sampling interval between inferencing')
    parser.add_argument('-debug', action='store_true', default=False, help='Debug flag')
    # parser.add_argument('-testmodel', action='store_true', default=False, help='Test with a fire image.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    run(args)