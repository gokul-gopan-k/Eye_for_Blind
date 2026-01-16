# Config and hyperparameters
from dataclasses import dataclass, field
from typing import FrozenSet
import torch

@dataclass(frozen=True)
class AppConfig:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_MODEL_PATH: str = "best_model.pt"
    MIDAS_MODEL_TYPE: str = "DPT_Large"
    NORMAL_LABELS: FrozenSet[str] = frozenset({"bus", "traffic sign", "bicycle", "person", "dog","motorcycle","tree","car"})
    CRITICAL_LABELS: FrozenSet[str] = frozenset({"electric pole", "manhole"})
    NO_OBJECT_COOLDOWN_SEC: float = 5.0
    YOLO_CONF_THRESHOLD: float = 0.15
    YOLO_IOU_THRESHOLD: float = 0.4
    PRIORITY_CONF_THRESHOLD: float = 0.5
    DEPTH_NEAR_PERCENTILE: float = 20.0
    DEPTH_FAR_PERCENTILE: float = 70.0
    CENTER_BIAS_ALPHA: float = 0.7
    CENTER_BIAS_BETA: float = 0.3
    SPATIAL_THRESHOLD: float = 0.33
    AUDIO_DIR: str = "audio_clips"
    OUTPUT_DIR: str = "outputs"

def load_config() -> AppConfig:
    return AppConfig()
