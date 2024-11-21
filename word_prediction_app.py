import streamlit as st
import cv2
import requests
import base64
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import numpy as np
import av

# URL of the FastAPI endpoint (Update with a public IP or URL)
api_url = "https://word-interpreter-app-373962339093.europe-west1.run.app/predict_word"

# Define VideoProcessor for frame processing
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""
        self.progress = 0
        self.last_sent_time = 0
        self.frame_interval = 1  # Set frame interval to 1 second to avoid overloading

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Only process frames at specified intervals
        current_time = time.time()
        if current_time - self.last_sent_time < self.frame_interval:
            return frame

        # Convert frame to numpy array
        frame_rgb = frame.to_ndarray(format="bgr24")
        _, buffer = cv2.imencode('.jpg', frame_rgb)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the payload
        payload = {"frame": frame_base64}

        # Send frame to the FastAPI server for prediction
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.result_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"
                self.progress = 100
            else:
                self.result_text = f"Error: Received status code {response.status_code}"
                self.progress = 0
        except Exception as e:
            self.result_text = f"Error: {e}"
            self.progress = 0

        self.last_sent_time = current_time
        return frame


def main():
    st.title("Real-time Sign Language Prediction with Webcam")

    # WebRTC configuration to improve connectivity
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # WebRTC streamer to access the webcam with VideoProcessor
    webrtc_ctx = webrtc_streamer(
        key="sign_language_prediction",
        mode=WebRtcMode.SENDRECV,  # SENDRECV mode to access the camera and receive processed frames
        rtc_configuration=rtc_configuration,  # Use RTC configuration for better connectivity
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,  # Disable audio stream
        },
    )

    # Create placeholders for results
    result_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Update UI elements based on video processor results
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        while webrtc_ctx.state.playing:
            video_processor = webrtc_ctx.video_processor
            if video_processor:
                result_placeholder.text(video_processor.result_text)
                progress_bar.progress(video_processor.progress)
            time.sleep(0.1)

if __name__ == "__main__":
    main()

