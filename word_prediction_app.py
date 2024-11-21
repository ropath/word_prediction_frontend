import streamlit as st
import cv2
import requests
import base64
import numpy as np
from PIL import Image
import time

# URL of the FastAPI endpoint (Update with a public IP or URL)
api_url = "https://your-fastapi-endpoint.com/predict_word"

def main():
    st.title("Real-time Sign Language Prediction")
    st.markdown("Press 'Start' to initiate the webcam and start predictions.")

    # Placeholder for the video frame and prediction result
    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    run = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        last_sent_time = 0
        frame_interval = 1  # Set frame interval to 1 second to avoid overloading

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the webcam. Please check your webcam settings.")
                break

            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in the Streamlit app
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Send a frame to the API every frame_interval seconds
            current_time = time.time()
            if current_time - last_sent_time > frame_interval:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Prepare the payload
                payload = {"frame": frame_base64}

                # Send frame to the FastAPI server for prediction
                try:
                    response = requests.post(api_url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        result_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"
                    else:
                        result_text = f"Error: Received status code {response.status_code}"
                except Exception as e:
                    result_text = f"Error: {e}"

                # Update the prediction result in the Streamlit app
                result_placeholder.markdown(f"**{result_text}**")
                last_sent_time = current_time

            # Stop the loop if the stop button is pressed
            if stop:
                break

        cap.release()
        frame_placeholder.empty()
        result_placeholder.empty()

if __name__ == "__main__":
    main()

