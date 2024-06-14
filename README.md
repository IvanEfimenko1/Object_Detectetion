# Object Detection

This project uses models to perform object detection on images and video. The implementation is done in a Jupyter Notebook using PyTorch.

## Table of Contents

- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Object Detection](#object-detection)
- [Video Processing](#video-processing)

## Installation

### Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Docker (optional, for containerized execution)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/YourUsername/YOLO_Object_Detection.git
    cd YOLO_Object_Detection
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If you prefer to use Docker, build the Docker container:

    ```bash
    docker-compose build
    ```

## Running the Notebook

1. Activate the virtual environment if not already activated:

    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the `Caggle_9.ipynb` notebook and run the cells sequentially to perform object detection on images and videos.

## Project Structure


- `Caggle_9.ipynb`: Jupyter notebook containing the implementation of object detection.

- `processed_dataset_ul.csv`: The dataset used for training and validation (if applicable).

## Object Detection

The object detection process includes:
- Loading images from URLs or local files
- Using the YOLOv5 model to detect objects in images
- Drawing bounding boxes around detected objects

Example code snippet for object detection:

```python
import torch
from PIL import Image
from io import BytesIO
import requests

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Function to load image from URL
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Function to perform detection
def detect_objects(image, model):
    results = model(image)
    return results.pandas().xyxy[0]

# Example usage
url = 'https://example.com/image.jpg'
image = load_image(url)
results = detect_objects(image, model)
print(results)
Video Processing
The video processing includes:

Loading video frames
Applying object detection to each frame
Saving the processed video with bounding boxes
Example code snippet for video processing:

python
Копировать код
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

# Function to convert frame to image
def frame_to_image(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function to convert image to frame
def image_to_frame(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Video paths
video_path = 'input_video.mp4'
output_path = 'output_video.mp4'

# Load video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process video frames
with tqdm(total=total_frames, desc="Processing video") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to image
        image = frame_to_image(frame)
        
        # Perform object detection
        results = detect_objects(image, model)
        
        # Draw bounding boxes
        for _, row in results.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Write processed frame to output video
        out.write(frame)
        pbar.update(1)

cap.release()
out.release()
print("Video processing complete")
Steps to Add README.md to GitHub