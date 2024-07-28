import cv2
import os
import tempfile
import streamlit as st
import numpy as np
from PIL import Image

# Create the test folder if it doesn't exist
output_folder = 'test'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the test_video folder if it doesn't exist
video_output_folder = 'test_video'
if not os.path.exists(video_output_folder):
    os.makedirs(video_output_folder)

# Function to get the next available file number
def get_next_file_number(folder):
    files = os.listdir(folder)
    numbers = [int(file.split('.')[0]) for file in files if file.split('.')[0].isdigit()]
    if numbers:
        return max(numbers) + 1
    else:
        return 1

# Path to the Haar Cascade file
cascade_path = 'haarcascade_frontalface_default.xml'

# Check if the Haar Cascade file exists
if not os.path.isfile(cascade_path):
    st.error(f'Haar Cascade file not found: {cascade_path}')
else:
    # Use Haar Cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Streamlit app
    st.title('🚀Face Detection App🚀')

    # Sidebar option to choose between image and video detection
    option = st.sidebar.selectbox("Choose Detection Type", ("Image", "Video"))

    if option == "Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Check the orientation of the image and rotate if necessary
            r=st.sidebar.radio("Do you want to routate the img?",('Yes','No'))
            if r=='Yes':
                if img.shape[0] > img.shape[1]:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                pass
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the original image in RGB
            st.image(img_rgb, caption='Original Image', use_column_width=True)

            if st.sidebar.button('Detect Faces'):
                # Convert the image to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Face detection
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around the detected faces
                img_with_faces = img.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Convert the image with faces to RGB
                img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)

                # Get the next available file number
                file_number = get_next_file_number(output_folder)
                output_path = os.path.join(output_folder, f'{file_number}.jpg')

                # Save the image in the test folder
                cv2.imwrite(output_path, img_with_faces)

                # Display the original and detected faces images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(img_with_faces_rgb, caption="Detected Faces", use_column_width=True)

                st.write(f"Saved image as: {output_path}")

    elif option == "Video":
        uploaded_video = st.sidebar.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            # Open the uploaded video file
            cap = cv2.VideoCapture(tfile.name)

            # Get the video properties
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Get the next available file number
            file_number = get_next_file_number(video_output_folder)
            video_output_path = os.path.join(video_output_folder, f'{file_number}.mp4')

            # Initialize the video writer
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Face detection
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Write the frame with detected faces to the output video
                out.write(frame)

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame with detected faces
                stframe.image(frame_rgb, use_column_width=True)

            cap.release()
            out.release()

            st.write(f"Saved video as: {video_output_path}")

            # Display the saved video
            st.video(video_output_path,format="video/mp4",start_time=0)
