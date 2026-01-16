# YOLO & MiDaS wrappers
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from utils import ensure_dir_exists
import logging
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)

class ObjectDetector:
    """Wrapper for the YOLOv8 object detection model."""
    def __init__(self, model_path: str, device: str):
        try:
            self.model = YOLO(model_path)
            self.device = device
            self.model.to(device)
            logger.info("YOLOv8 model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load YOLOv8 model: %s", e)
            raise

    def detect(self, img_path: str, conf: float,iou: float):
        detections = []
        try:
            results_list = self.model.predict(
                img_path, conf=conf,iou = iou, device=self.device, verbose=False
            )
            yolo_result = results_list[0]
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            classes = yolo_result.boxes.cls.cpu().numpy()
            confs = yolo_result.boxes.conf.cpu().numpy()

            if len(boxes) == 0:
                return [], yolo_result

            for box, cls, conf_val in zip(boxes, classes, confs):
                detections.append(
                    {
                        "box": box,
                        "label": self.model.names[int(cls)],
                        "confidence": float(conf_val),
                    }
                )
            return detections, yolo_result
        except Exception as e:
            logger.error("Error during object detection: %s", e)
            return [], None

class DepthEstimator:
    def __init__(self, model_type: str, device: str):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).eval().to(device)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate(self, image, near_p: float, far_p: float):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_transformed = self.transform(rgb_image).to(self.device)
        with torch.no_grad():
            depth = self.model(input_transformed)
        depth_map = depth.squeeze().cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max != d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        near_th, far_th = np.percentile(depth_map, [near_p, far_p])
        return depth_map, near_th, far_th

def load_models(config):
    detector = ObjectDetector(config.YOLO_MODEL_PATH, config.DEVICE)
    estimator = DepthEstimator(config.MIDAS_MODEL_TYPE, config.DEVICE)
    return detector, estimator
