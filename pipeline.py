# AssistivePipeline class (priority scoring, spatial analysis)

from utils import get_audio_id
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class AssistivePipeline:
    def __init__(self, config):
        self.config = config
        self.critical_labels = config.CRITICAL_LABELS
        self.normal_labels = config.NORMAL_LABELS
        self.alpha = config.CENTER_BIAS_ALPHA
        self.beta = config.CENTER_BIAS_BETA
        self.spatial_threshold = config.SPATIAL_THRESHOLD
        self.type_weight = 10.0
        self.distance_weight = 3.0
        self.spatial_weight = 1.0
        self.english_dict = {}  
        self.hindi_dict = {}    

    def compute_priority_score(self, label, distance_category, spatial_position):
        type_score = 1.0 if label.lower() in self.critical_labels else 0.0
        distance_score = {"right in front of you":3,"a few steps ahead":2,"some distance ahead":1}.get(distance_category,0)
        spatial_score = {"In front of you,":1,"On your left,":0.5,"On your right,":0.5}.get(spatial_position,0)
        return self.type_weight*type_score + self.distance_weight*distance_score + self.spatial_weight*spatial_score

    def _get_depth_for_box(self, depth_map, box):
        x1, y1, x2, y2 = map(int, box)
        region = depth_map[y1:y2, x1:x2]
        return np.median(region) if region.size>0 else 1.0

    def _classify_distance(self, depth, near_th, far_th):
        if depth > far_th: return "right in front of you"
        if depth > near_th: return "a few steps ahead"
        return "some distance ahead"

    def _get_spatial_direction(self, box, img_width):
        x1, _, x2, _ = box
        x_center = (x1+x2)/2
        if x_center < img_width*self.spatial_threshold: return "On your left,"
        if x_center > img_width*(1-self.spatial_threshold): return "On your right,"
        return "In front of you,"

    def prioritize_objects(self, detections: List[Dict[str, Any]], depth_map, near_th, far_th) -> Optional[Tuple[str,str,str]]:
        if len(detections) == 0: return None
        scored = []
        for det in detections:
            label = det["label"].lower()
            depth_val = self._get_depth_for_box(depth_map, det["box"])
            distance_category = self._classify_distance(depth_val, near_th, far_th)
            spatial_position = self._get_spatial_direction(det["box"], depth_map.shape[1])
            score = self.compute_priority_score(label, distance_category, spatial_position)
            scored.append((score,label,distance_category,spatial_position))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][2], scored[0][3]
