import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import os

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify, count, and trigger alerts in real-time.")

# Sidebar configurations
st.sidebar.header("Configuration Settings")
target_object = st.sidebar.selectbox("Select object to track:", list(model.names.values()))
alert_threshold = st.sidebar.slider("Alert Threshold (Min instances):", 1, 10, 1)

# Session state initialization
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'alert' not in st.session_state:
    st.session_state.alert = False

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.save_frame_flag = False
        self.count = 0
        self.alert = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 tracking
        results = self.model.track(
            img,
            persist=True,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        #Object Counting
        count = 0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]
                if class_name == target_object:
                    count += 1
        self.count = count

        #Triggering alerts
        if self.count >= alert_threshold:
            self.alert = True
        else:
            self.alert = False

        #Saving detected frames
        if self.save_frame_flag:
            os.makedirs("saved_frames", exist_ok=True)
            cv2.imwrite("saved_frames/detected_frame.jpg", annotated_frame)
            self.save_frame_flag = False  # Reset flag

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Start WebRTC streamer
ctx = webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    async_processing=True,  # smoother performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# Update session state from the video processor
if ctx.video_processor:
    st.session_state.count = ctx.video_processor.count
    st.session_state.alert = ctx.video_processor.alert

# UI Displays
st.write(f"### Object Count (Target: **{target_object}**): {st.session_state.count}")

if st.session_state.alert:
    st.warning(f"⚠️ ALERT: Threshold exceeded! Detected {st.session_state.count} {target_object}(s).")

# Save frame button
if st.button("Save Current Frame"):
    if ctx.video_processor:
        ctx.video_processor.save_frame_flag = True
        st.success("Frame saved to 'saved_frames/detected_frame.jpg'")
    else:
        st.warning("Please start the video stream first.")