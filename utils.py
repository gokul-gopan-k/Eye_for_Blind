# Helper functions (audio mapping, depth visualization)
import os
import matplotlib.pyplot as plt
import numpy as np
import constants

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

def create_depth_map_fig(depth_map: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10,8))
    cax = ax.imshow(depth_map, cmap='inferno')
    fig.colorbar(cax, label='Normalized Depth')
    ax.axis('off')
    return fig

def get_audio_id(label:str, distance:str, spatial:str):
    obj_map = {"bus":1,"manhole":2,"person":3,"dog":4}
    dist_map = {"right in front of you":0,"a few steps ahead":1,"some distance ahead":2}
    spat_map = {"In front of you,":0,"On your left,":1,"On your right,":2}
    try:
        return int(f"{obj_map.get(label,9)}{dist_map.get(distance,0)}{spat_map.get(spatial,0)}")
    except:
        return None

def predefined_text(audio_id):
    try:
        return constants.english_dict[audio_id], constants.hindi_dict[audio_id]
    except KeyError:
        return "English text not found", "Hindi text not found"
