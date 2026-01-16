# Eye_for_Blind

Real-time vision-based hazard detection and guidance system for visually impaired users.
This project integrates YOLOv8 object detection and MiDaS depth estimation to provide context-aware audio alerts in both English and Hindi, helping users navigate safely through real-world environments.
The system is designed as a production-ready pipeline, demonstrating practical experience with computer vision, deep learning, model optimization, and real-time inference.

| Input Image |  Output |
|------------|-------------|
| ![Input](sample_output/input_image.png) | ![Output](sample_output/yolo_output.png) |


## Features

- End-to-end vision pipeline: Detect objects → Estimate distance → Prioritize hazards → Generate audio alerts.
- Object Detection: YOLOv8 with advanced augmentations for high accuracy.
- Depth Estimation: MiDaS DPT_Large for spatial awareness and distance categorization.
- Priority Scoring: Critical objects like manholes and electric poles are prioritized based on distance and spatial position.
- Audio Feedback: Context-aware alerts in English and Hindi.
- Interactive Visualization: Shows annotated object detections and depth maps.
- Production-ready Design: Handles edge cases, temporary files, and robust error handling.

##  Demo Workflow

- Upload an image.
- YOLOv8 detects objects in the scene.
- MiDaS estimates depth for spatial positioning.
- Assistive pipeline prioritizes hazards and generates audio alert.
- Visualizations of detected objects and depth maps are displayed.

## Installation


1. **Clone the repository:**
```bash
git clone https://github.com/Eye_for_Blind/assistive-pipeline.git
cd assistive-pipeline
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Usage**
```bash
Run the Streamlit app:
streamlit run app.py
```

- Upload an image (JPG/PNG) in the sidebar.
- Click Analyze Image and Generate Audio.
- Listen to English/Hindi audio alerts and view visual detection/depth maps.
- Audio Files: Place .wav files in audio_clips/ corresponding to your object IDs.

## Key Technical Highlights

- Computer Vision & Deep Learning:YOLOv8 (object detection),MiDaS (depth estimation),PyTorch-based training & inference
- Data Handling:Image preprocessing, bounding box evaluation,Prioritization of critical objects based on distance and spatial positioning
- Real-world Deployment Readiness:Temporary output management, exception handling.Scalable for edge deployment (real-time inference capable)
-Evaluation & Robustness:
Handles edge cases: multiple objects, missing audio, depth extremes
Priority scoring combines object type, distance, and spatial bias

# Performance metrics
The object detection model (YOLOv8) used in this pipeline was trained on a custom dataset of common urban objects relevant to assistive navigation for the visually impaired. Below are the validation metrics:

| Class          | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|----------------|---------------|------------|-------|-----------|
| Bus            | 0.971         | 0.941      | 0.967 | 0.919     |
| Dog            | 0.940         | 0.928      | 0.945 | 0.863     |
| Manhole        | 0.956         | 0.786      | 0.887 | 0.676     |
| Electric Pole  | 0.675         | 0.434      | 0.588 | 0.366     |
| Person         | 0.766         | 0.684      | 0.769 | 0.569     |
| Bicycle        | 0.829         | 0.846      | 0.875 | 0.827     |
| Car            | 0.702         | 0.747      | 0.782 | 0.661     |
| Motorcycle     | 0.856         | 0.910      | 0.926 | 0.852     |
| Traffic Sign   | 0.582         | 0.612      | 0.630 | 0.548     |
| Tree           | 0.699         | 0.763      | 0.785 | 0.691     |


## Key Insights

- High performance on common hazards: Objects such as bus, dog, and manhole achieve both high precision and recall, ensuring reliable detection in typical urban scenarios.
- Critical hazard prioritization: The pipeline flags electric poles and manholes as critical, even if detection confidence is moderate.
- Current limitations: Recall for electric poles (~43%) is lower than ideal, highlighting a potential area for model improvement to further enhance user safety.
- End-to-end assistive pipeline: Despite some detection limitations, the system combines object detection, depth estimation, spatial prioritization, and multimodal (audio + visual) feedback, providing a practical tool for visually impaired users.

## Future Improvements
- Expand audio library for more objects and dynamic speech synthesis.
- Incorporate user-customizable priorities and thresholds via GUI.
