# Face Detection with Image and Video Options

This project is part of my internship at CODSOFT, where I developed a face detection web application using OpenCV and Streamlit. The app allows users to upload images or videos, detect faces, and display the results.

## Features

- **Image Upload and Detection**:
  - Users can upload an image.
  - The app detects faces and draws rectangles around them.
  - The processed image is saved and displayed alongside the original.

- **Video Upload and Detection**:
  - Users can upload a video.
  - The app processes each frame to detect faces.
  - The processed video is saved and can be played back within the app.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/face-detection-app.git
    cd face-detection-app
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the Haar Cascade file for face detection:
    - [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
    - Place it in the project directory.

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501`.

3. Use the sidebar to upload an image or a video.

4. Click the "Detect Faces" button to start the face detection process.

