import streamlit as st
import cv2
import requests
import base64
import time
import tempfile
from PIL import Image
import numpy as np

# URL of the FastAPI endpoint
api_url = "https://word-interpreter-app-373962339093.europe-west1.run.app/predict_word"

def main():
    st.title("Sign Language Prediction from Video Upload")
    st.text("Upload a short video clip for sign language prediction.")

    # File uploader for video
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return

        # Display the uploaded video
        st.video(uploaded_video)

        placeholder = st.empty()
        gif_placeholder = st.empty()
        progress_bar = st.progress(0)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to JPEG format and encode it to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare the payload
            payload = {"frame": frame_base64}

            try:
                # Send POST request to the API
                response = requests.post(api_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prediction_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"

                    # Check if GIF is included in the response
                    if 'gif' in result:
                        gif_base64 = result['gif']
                        gif_bytes = base64.b64decode(gif_base64)
                        gif_placeholder.image(gif_bytes)
                else:
                    prediction_text = f"Error: Received status code {response.status_code}"
            except Exception as e:
                prediction_text = f"Error: {e}"

            # Display the frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            placeholder.image(img_pil)

            # Update the progress bar
            progress += 1
            progress_bar.progress(progress / frame_count)

            # Add delay to control the rate of requests to avoid overwhelming the backend
            time.sleep(0.05)

        # Release the video resource
        cap.release()
        st.success("Video processing complete.")

if __name__ == "__main__":
    main()

