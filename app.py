from __future__ import print_function

import numpy as np
import time
import argparse
import cv2
import datetime
import os
import torch
import torch.nn as nn

from models.experimental import Ensemble
from models.common import Conv, DWConv
from utils.general import non_max_suppression, apply_classifier

from waggle.plugin import Plugin

class YOLOv7_Main():
    def __init__(self, args, weightfile):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

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


    def run(self, frame, args):
        sized = cv2.resize(frame, (640, 640))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        image = sized / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image)[0]

        return pred

def run(args):
  with Plugin() as plugin:
    yolo = YOLOv7_Main(args)

    if args.input.lower() == 'camera':
        cap = cv2.VideoCapture(0)
    elif args.input.startswith(('http://', 'https://', 'rtsp://')):
        cap = cv2.VideoCapture(args.input)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: File '{args.input}' does not exist.")
            return
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = yolo.run(frame, args)

        # Process results (e.g., draw bounding boxes)
        detections = []
        for det in results:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    label = f'{args.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detections.append(label)

        # Save the frame with detections
        output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)

        # Print information in the terminal
        print(f"Frame {frame_count}: Saved to {output_path}")
        if detections:
            print("Detections:", ", ".join(detections))
        else:
            print("No detections in this frame")
        print("-" * 50)

        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plugin.publish("log", f"Processing complete. {frame_count} frames saved in {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO v7 Detection')
    parser.add_argument('-weight', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('-input', type=str, default='camera', help='source (camera/url/video file path)')
    parser.add_argument('-conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('-iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('-output-dir', type=str, default='output', help='directory to save output frames')
    return parser.parse_args()
  

if __name__ == '__main__':
    args = parse_args()
    run(args)