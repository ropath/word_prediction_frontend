import streamlit as st
import cv2
import requests
import base64
import time
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
from aiortc.contrib.media import MediaPlayer

# URL of the FastAPI endpoint (Update with a public IP or URL)
api_url = "https://your-fastapi-endpoint.com/predict_word"

# Define VideoProcessor for frame processing
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""
        self.last_sent_time = 0
        self.frame_interval = 1  # Set frame interval to 1 second to avoid overloading

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
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
            else:
                self.result_text = f"Error: Received status code {response.status_code}"
        except Exception as e:
            self.result_text = f"Error: {e}"

        self.last_sent_time = current_time
        return frame


def main():
    st.title("Real-time Sign Language Prediction")
    st.markdown("Press 'Start' to initiate the webcam and start predictions.")

    # WebRTC streamer to access the webcam with VideoProcessor
    webrtc_ctx = webrtc_streamer(
        key="sign_language_prediction",
        mode=WebRtcMode.SENDRECV,  # SENDRECV mode to access the camera and receive processed frames
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,  # Disable audio stream
        },
        async_processing=True,
    )

    # Display results
    result_placeholder = st.empty()

    if webrtc_ctx and webrtc_ctx.video_processor:
        while webrtc_ctx.state.playing:
            video_processor = webrtc_ctx.video_processor
            if video_processor:
                result_placeholder.markdown(f"**{video_processor.result_text}**")
            time.sleep(0.1)

    # Restart button for the app
    if st.button("Restart Application"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()


