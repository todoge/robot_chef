import torch
import torchvision
import numpy as np
from torchvision.transforms import functional as F

OBJ_DETECTION_LABELS = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

class Object_Detector:
    def __init__(self): 
      self.obj_detection_model = self.setup()

    def setup(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obj_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        obj_detection_model.to(device)
        obj_detection_model.eval()

        return obj_detection_model

    def rgb_to_img_tensor(self, rgb_img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rgb_float = rgb_img.astype(np.float32) / 255.0
        img_tensor = F.to_tensor(rgb_float).to(device)
        return img_tensor

    def detect_object(self, rgb_img, threshold=0): #threshold is confidence cutoff
        img_tensor = self.rgb_to_img_tensor(rgb_img)
        with torch.no_grad():
            detections = self.obj_detection_model([img_tensor])[0]
        pred_scores = detections['scores'].cpu().numpy()
        scores = pred_scores[pred_scores >= threshold]
        pred_bboxes = detections['boxes'].cpu().numpy()
        boxes = pred_bboxes[pred_scores >= threshold].astype(np.int32)
        labels = detections['labels'][:len(boxes)]
        pred_classes = [OBJ_DETECTION_LABELS[i] for i in labels.cpu().numpy()]
        bbox = None
        for box, label, score in zip(boxes, pred_classes, scores):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box.astype(int)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            bbox = (y1,y2,x1,x2)
            print(f"Detected object {label} with score {score:.2f} at {x1},{y1},{x2},{y2}")
            break # to take take the highest confidence right now cause the model is detecting rubbish
        print("bbox: ", bbox)
        return bbox

