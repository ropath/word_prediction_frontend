import streamlit as st
import cv2
import requests
import base64
import time
import tempfile
from PIL import Image

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

        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("End of video file.")
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
            placeholder.image(img_pil, use_container_width=True)

            # Update the progress bar while collecting frames for prediction
            progress = min(progress + 1, total_frames)
            progress_bar.progress(progress / total_frames)

            # Display the prediction result
            st.success(prediction_text)

            # Add delay to control the rate of requests to avoid overwhelming the backend
            time.sleep(0.001)

        # Release the video resource
        cap.release()

if __name__ == "__main__":
    main()

