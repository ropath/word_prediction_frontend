import streamlit as st
import cv2
import requests
import base64
import time
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import numpy as np
import av
import streamlit.components.v1 as components

# URL of the FastAPI endpoint (Update with a public IP or URL)
api_url = "https://your-fastapi-endpoint.com/predict_word"

# Define VideoProcessor for frame processing
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""
        self.progress = 0
        self.last_sent_time = 0
        self.frame_interval = 0.5  # Reduced interval between frames to send (in seconds)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        if current_time - self.last_sent_time < self.frame_interval:
            return frame

        frame_rgb = frame.to_image()  # Convert to PIL Image
        frame_np = np.array(frame_rgb)  # Convert to numpy array
        _, buffer = cv2.imencode('.jpg', frame_np)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the payload
        payload = {"frame": frame_base64}

        try:
            # Send POST request to the API
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                self.result_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"
                
                # Update progress bar if prediction is in progress
                if result['prediction'] == "collecting frames for prediction...":
                    self.progress = min(self.progress + 1, 100)
                else:
                    self.progress = 100
            else:
                self.result_text = f"Error: Received status code {response.status_code}"
        except Exception as e:
            self.result_text = f"Error: {e}"

        self.last_sent_time = current_time
        return frame


def main():
    st.title("Real-time Sign Language Prediction with Webcam")
    st.text("Press 'Start' to initiate the webcam and start predictions.")

    # WebRTC streamer to access the webcam with VideoProcessor
    webrtc_ctx = webrtc_streamer(
        key="sign_language_prediction",
        mode=WebRtcMode.SENDRECV,  # SENDRECV mode to access the camera and receive processed frames
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 640, "max": 1280},
                "height": {"min": 480, "ideal": 480, "max": 720},
                "frameRate": {"ideal": 15, "max": 30}
            },
            "audio": False,  # Disable audio stream
        },
    )

    # Create placeholders for results
    result_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Use callbacks to display prediction results
    if webrtc_ctx and webrtc_ctx.state is not None and webrtc_ctx.video_processor:
        while webrtc_ctx.state.playing:
            # Update prediction text
            if webrtc_ctx.video_processor:
                result_placeholder.text(webrtc_ctx.video_processor.result_text)

                # Update the progress bar
                progress_bar.progress(webrtc_ctx.video_processor.progress / 100)

            # Add a small delay to prevent overwhelming the interface
            time.sleep(0.1)

    # Adding HTML/JavaScript for a more interactive UI
    components.html(
        """
        <div style="text-align: center; margin-top: 20px;">
            <button onclick="document.location.reload()" style="padding: 10px 20px; font-size: 16px;">Restart Application</button>
        </div>
        """,
        height=100,
    )

if __name__ == "__main__":
    main()

