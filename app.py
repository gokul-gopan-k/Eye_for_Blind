# This the main streamlit app code
import streamlit as st
import os
from config import load_config
from models import load_models
from pipeline import AssistivePipeline
from utils import ensure_dir_exists, create_depth_map_fig, get_audio_id, predefined_text
import cv2
import constants
def main_app():
    st.set_page_config(page_title="Assistive Pipeline", layout="wide")
    st.title(" Assistive Pipeline for the Visually Impaired")

    # Load config and models
    config = load_config()
    detector, estimator = load_models(config)
    st.sidebar.success("All models loaded successfully!")
    st.sidebar.info(f"Running on device: **{config.DEVICE.upper()}**")

    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose a JPG or PNG image", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload an image using the sidebar to begin.")
        return

    ensure_dir_exists(config.OUTPUT_DIR)
    temp_img_path = os.path.join(config.OUTPUT_DIR, uploaded_file.name)
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = cv2.imread(temp_img_path)

    if image is None:
        st.error("Could not read uploaded image. It might be corrupt.")
        return

    st.header("1. Your Uploaded Image")
    st.image(uploaded_file, caption="Image to be analyzed", use_container_width=True)

    if st.button(" Analyze Image and Generate Audio", type="primary"):
        pipeline = AssistivePipeline(config)
        detections, yolo_result = detector.detect(temp_img_path, conf =config.YOLO_CONF_THRESHOLD,iou =config.YOLO_IOU_THRESHOLD)
        depth_map, near_th, far_th = estimator.estimate(image, config.DEPTH_NEAR_PERCENTILE, config.DEPTH_FAR_PERCENTILE)
        result = pipeline.prioritize_objects(detections, depth_map, near_th, far_th)

        if result:
            label, dist, spatial = result
            audio_id = get_audio_id(label, dist, spatial)
            if audio_id in constants.english_dict:
                eng_text, hin_text = predefined_text(audio_id)
                st.subheader("🗣️ Audio Description")
                st.markdown(f"**English:** `{eng_text}`")
                st.markdown(f"**Hindi:** `{hin_text}`")
                audio_path = os.path.join(config.AUDIO_DIR, f"{audio_id}.wav")
                if os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                else:
                    st.warning(f"Audio file {audio_id}.wav not found in {config.AUDIO_DIR}")
            else:
                st.error(f"No predefined text/audio found for ID: {audio_id}")
        else:
            st.info("Environment is clear.")

        # Visuals
        st.subheader(" Visual Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Object Detections (YOLOv8)**")
            if yolo_result:
                annotated_img = yolo_result.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True)
        with col2:
            st.markdown("**Depth Map (MiDaS)**")
            fig = create_depth_map_fig(depth_map)
            st.pyplot(fig)

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

if __name__ == "__main__":
    main_app()
