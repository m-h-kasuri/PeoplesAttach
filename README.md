import os
import time
import requests
import cv2
import torch
import numpy as np
from yt_dlp import YoutubeDL
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

# Initialize YOLOv8 for person detection
yolo_model = YOLO("yolov8n.pt")

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Load pre-trained FaceNet model


def download_image_from_url(url):
    """
    Downloads an image from a URL and converts it to a format usable by OpenCV.

    Parameters:
    - url: The URL of the image.

    Returns:
    - frame: The image as a NumPy array (usable with OpenCV).
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        image_data = np.asarray(bytearray(response.content), dtype="uint8")
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def download_video_with_progress(url, output_dir="downloads"):
    """
    Downloads a video from a URL using yt-dlp, showing progress and skipping if the file already exists.

    Parameters:
    - url: The URL of the video to download.
    - output_dir: Directory to save the downloaded video.

    Returns:
    - Path to the downloaded video file.
    """
    os.makedirs(output_dir, exist_ok=True)

    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            speed = d.get('speed', 0)
            progress = (downloaded / total) * 100 if total > 0 else 0
            remaining_time = (total - downloaded) / speed if speed > 0 else 0
            remaining_minutes, remaining_seconds = divmod(remaining_time, 60)
            print(
                f"\rDownloading: {progress:.2f}% complete | "
                f"Speed: {speed / 1024:.2f} KB/s | "
                f"Remaining time: {int(remaining_minutes):02d}:{int(remaining_seconds):02d}",
                end=""
            )
        elif d['status'] == 'finished':
            print("\nDownload complete!")

    # Get video information without downloading
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info['title']
        video_ext = 'mp4'
        output_file = os.path.join(output_dir, f"{video_title}.{video_ext}")

    # Check if the video already exists
    if os.path.exists(output_file):
        print(f"Video already exists: {output_file}")
        return output_file

    # Download the video if it does not exist
    ydl_opts = {
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "format": "mp4",
        "quiet": True,
        "progress_hooks": [progress_hook],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_file

# Function to extract embeddings for a target image
def extract_person_embeddings(frame, boxes, model):
    """
    Extract embeddings for persons detected in the frame.

    Parameters:
    - frame: The current video frame (numpy array).
    - boxes: List of bounding boxes for detected persons [[x1, y1, x2, y2], ...].
    - model: The pre-trained FaceNet model used for embedding extraction.

    Returns:
    - embeddings: A list of embeddings for the detected persons.
    """
    embeddings = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Crop the detected person region from the frame
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size > 0:  # Ensure the crop is valid
            # Resize the crop to the required size for the model
            resized = cv2.resize(person_crop, (160, 160))
            
            # Convert the resized crop to a PyTorch tensor
            tensor = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
            tensor = (tensor - 127.5) / 128.0  # Normalize the input
            
            # Extract the embedding using the model
            with torch.no_grad():
                embedding = model(tensor).detach().cpu().numpy()
            embeddings.append(embedding)
    return embeddings

# Function to process a single frame
def process_frame(frame_data):
    """
    Process a single frame to detect persons and match them to target embeddings.
    """
    frame, frame_idx, target_embeddings, resnet, accuracy = frame_data
    frame = cv2.resize(frame, (640, 360))

    # Detect persons using YOLOv8
    results = yolo_model.predict(frame, classes=[0])  # Detect only persons (class 0)
    detections = []

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():  # Extract bounding boxes
            if box.shape[0] >= 4:  # Handle coordinates and optional confidence
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append([x1, y1, x2, y2])

    # Match detections to target embeddings
    frame_clips = {name: [] for name in target_embeddings.keys()}
    for name, target_embedding in target_embeddings.items():
        embeddings = extract_person_embeddings(frame, detections, resnet)
        if embeddings:
            similarities = [cosine_similarity(e, target_embedding) for e in embeddings]
            if max(similarities) >= accuracy:
                frame_clips[name].append(frame)
    return frame_idx, frame_clips


# Main function to process video with multiprocessing
def process_video_with_multiprocessing(input_video, target_embeddings, resnet, accuracy=0.85, num_workers=2):
    """
    Process a video to detect target persons using multiprocessing for efficiency.
    """
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Processing video: {input_video}")
    print(f"Total frames: {total_frames}, Frame rate: {frame_rate} FPS")

    # Initialize timers and progress trackers
    start_time = time.time()
    frame_data = []
    frame_idx = 0

    # Collect frame data for processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_data.append((frame, frame_idx, target_embeddings, resnet, accuracy))
        frame_idx += 1

    cap.release()

    # Use multiprocessing to process frames
    print(f"Starting multiprocessing with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_frame, frame_data)

    # Combine results from all processes
    clips = {name: [] for name in target_embeddings.keys()}
    processed_frames = 0

    print(f"Processing frames... Total frames: {total_frames}")
    for _, frame_clips in results:
        processed_frames += 1
        for name in frame_clips.keys():
            clips[name].extend(frame_clips[name])

        # Log progress
        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / processed_frames
        remaining_frames = total_frames - processed_frames
        remaining_time = avg_time_per_frame * remaining_frames
        remaining_minutes, remaining_seconds = divmod(remaining_time, 60)

        print(
            f"\rProcessed frames: {processed_frames}/{total_frames} "
            f"({processed_frames/total_frames:.2%}) | "
            f"Elapsed time: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | "
            f"Remaining time: {int(remaining_minutes):02d}:{int(remaining_seconds):02d}",
            end=""
        )

    print("\nProcessing complete!")
    return clips


# Save extracted clips
def save_clips(clips, output_dir="output_clips", frame_rate=30):
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for person_name, frames in clips.items():
        person_folder = os.path.join(output_dir, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Save clip as a video
        output_file = os.path.join(person_folder, f"{person_name}_clip.mp4")
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (640, 360))
        for frame in frames:
            out.write(frame)
        out.release()
        saved_files.append(output_file)

    return saved_files

# Main function
if __name__ == "__main__":
    # Step 1: Download video
    video_url = "https://www.youtube.com/watch?v=z_NK9qRMZ5Q&pp=ygUMc25vb2tlciAyMDI0"
    print('working on downloading video')
    downloaded_video = download_video_with_progress(video_url)
    print("downloaded video completed")
    # Step 2: Define image URLs for target persons
    # target_images = {
    #     "Person1": "data:image/webp;base64,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",  # Replace with actual URL
    #     "Person2": "data:image/webp;base64,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",  # Replace with actual URL
    # }

    # # Initialize the FaceNet ResNet model for embeddings
    # from facenet_pytorch import InceptionResnetV1

    # resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # # Extract embeddings for target persons
    # target_embeddings = {}
    # for name, image_url in target_images.items():
    #     # Download image from URL
    #     target_frame = download_image_from_url(image_url)
    #     if target_frame is None:
    #         print(f"Skipping {name} due to failed image download.")
    #         continue

    #     # Extract embedding for the target image
    #     embedding = extract_person_embeddings(
    #         target_frame, [[0, 0, target_frame.shape[1], target_frame.shape[0]]], resnet
    #     )
    #     if embedding:
    #         target_embeddings[name] = embedding[0]
    #     else:
    #         print(f"Could not extract embedding for {name}.")

    target_images = {
        "Person1": "./trump.png",  # Image of the first person
        "Person2": "./ronnie.png",  # Image of the second person
    }

    # Initialize the FaceNet ResNet model for embeddings
    from facenet_pytorch import InceptionResnetV1

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Extract embeddings for target persons
    target_embeddings = {}
    for name, image_path in target_images.items():
        target_frame = cv2.imread(image_path)
        target_embeddings[name] = extract_person_embeddings(
            target_frame, [[0, 0, target_frame.shape[1], target_frame.shape[0]]], resnet
        )[0]

    # Process video with multiprocessing
    clips = process_video_with_multiprocessing(downloaded_video, target_embeddings, resnet, accuracy=0.85)

    # Save the extracted clips
    saved_clips = save_clips(clips)
