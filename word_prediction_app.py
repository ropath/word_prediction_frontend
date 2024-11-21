import streamlit as st
import cv2
import requests
import base64
import time
import tempfile
from PIL import Image, ImageSequence

# URL of the FastAPI endpoint
api_url = "https://word-interpreter-app-373962339093.europe-west1.run.app/predict_word"

def main():
    st.title("Sign Language Prediction from Uploaded Video")
    st.text("Upload a video file to start predictions.")

    # File uploader to upload video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    placeholder = st.empty()
    gif_placeholder = st.empty()
    progress_bar = st.progress(0)

    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return

        progress = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_base64 = []
        pil_frames = []

        # Read all frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("End of video file.")
                break

            # Resize the frame to reduce size
            frame = cv2.resize(frame, (320, 240))

            # Convert frame to JPEG format and encode it to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)

            # Convert frame to RGB and save to PIL frames list for GIF creation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_frames.append(pil_frame)

            # Display the frame in Streamlit
            placeholder.image(pil_frame)

            # Update the progress bar
            progress += 1
            progress_bar.progress(progress / total_frames)
            time.sleep(0.02)

        # Send all frames to the API at once
        payload = {"frames": frames_base64}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                prediction_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"
                st.success(prediction_text)
            else:
                st.error(f"Error: Received status code {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

        # Create GIF from the PIL frames
        if pil_frames:
            gif_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            pil_frames[0].save(
                gif_buffer,
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,  # Endless loop
                duration=50  # Frame duration in milliseconds
            )
            gif_buffer.seek(0)
            gif_placeholder.image(gif_buffer.name)

        # Release the video resource
        cap.release()

if __name__ == "__main__":
    main()



