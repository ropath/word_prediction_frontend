import streamlit as st
import cv2
import requests
import base64
import time
from PIL import Image

# URL of the FastAPI endpoint
api_url = "https://word-interpreter-app-373962339093.europe-west1.run.app/predict_word"

def main():
    st.title("Real-time Sign Language Prediction with Webcam")
    st.text("Press 'Start' to initiate the webcam and start predictions.")

    # Buttons to control the webcam
    run = st.button("Start")
    stop = st.button("Stop")
    placeholder = st.empty()
    gif_placeholder = st.empty()
    progress_bar = st.progress(0)

    if run:
        # Open the webcam using the provided stream URI
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open video stream.")
            return

        progress = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from video stream.")
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
            placeholder.image(img_pil, use_container_width=True) # caption=prediction_text

            # Update the progress bar while collecting frames for prediction
            if result['prediction'] == "collecting frames for prediction...":
                progress = min(progress + 1, 100)
                progress_bar.progress(progress / 100)
            else:
                st.success(prediction_text)
                break

            # Add delay to control the rate of requests to avoid overwhelming the backend
            time.sleep(0.001)

            # Stop the loop if the stop button is pressed
            if stop:
                break

        # Release the webcam resource
        cap.release()

if __name__ == "__main__":
    main()
