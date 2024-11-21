import streamlit as st
import tempfile
from moviepy.editor import VideoFileClip

def main():
    st.title("Sign Language Prediction from Uploaded Video as GIF")
    st.text("Upload a video file to convert it into an animated GIF.")

    # File uploader to upload video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        try:
            # Use moviepy to convert the video to a GIF
            video_clip = VideoFileClip(tfile.name)

            # Set a reasonable duration or resize if the video is too large
            if video_clip.duration > 10:
                st.warning("Video is longer than 10 seconds. Trimming to the first 10 seconds.")
                video_clip = video_clip.subclip(0, 10)

            video_clip = video_clip.resize(height=240)  # Resize to reduce size if necessary

            # Save GIF to a temporary file
            gif_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            video_clip.write_gif(gif_file.name, fps=10)  # Reduce fps to make the GIF size smaller

            # Display the generated GIF in Streamlit
            st.image(gif_file.name, caption="Generated GIF from Uploaded Video")#, use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()


