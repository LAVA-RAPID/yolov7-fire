# YOLOv7-Fire

YOLOv7-Fire is a fire detection model based on YOLOv7. It is finetuned on the [DFire Dataset](https://github.com/gaiasd/DFireDataset). The DFire Dataset contains a total of 21,527 images and labels: 17,221 images and labels are in the training directory, and 4,306 are in the testing directory.


## Arguments
```
-weight: model.pt path (default: yolov7-fire.pt)

-stream: ID or name of a stream (default: bottom_camera)

-conf-thres: object confidence threshold (default: 0.40)

-iou-thres: IOU threshold for NMS (default: 0.45)

-continuous: Run plugin forever (default: False)

-sampling-interval: Sampling interval between inferencing before uploading an image (default: 1)

-debug: Debug mode (default: False)

-test: Tests model on a fire image (default: False)
```

## Data Fetching

Data can be obtained from `env.detection`, which shows `{fire: <count>, smoke: <count>}`.




