import os
import cv2
import concurrent.futures
import shutil
from insightface.app import FaceAnalysis
import numpy as np

# Initialize the InsightFace app for CPU usage
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU (ctx_id=-1)

# Define folder paths
dataset_path = os.path.join(os.getcwd(), "dataset")  # Directory with images to scan
training_image_path = os.path.join(os.getcwd(), "training image")  # Directory with the user images to train
output_directory = os.path.join(os.getcwd(), "output")  # Output folder

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load and encode the user face image
def load_user_images(training_image_path):
    user_embeddings = []
    # Process each image in the training directory
    for file in os.listdir(training_image_path):
        if file.endswith((".jpg", ".png")):
            img_path = os.path.join(training_image_path, file)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Training image not found or unable to load: {img_path}")
            
            faces = app.get(img)
            if faces:
                user_embeddings.append(faces[0].embedding)  # Append embeddings for all user images
            else:
                print(f"No face found in training image: {img_path}")
    if not user_embeddings:
        raise Exception("No faces found in any training images")
    
    # Return a list of user embeddings
    return user_embeddings

user_embeddings = load_user_images(training_image_path)

# Downscale image for faster processing
def downscale_image(image_path, scale=0.5):
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {image_path}")

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height))

# Process a single image and check for matching faces
def process_image(image_path, user_embeddings, threshold=0.6):
    try:
        # Step 1: Downscale image
        scaled_image = downscale_image(image_path, 0.5)
        
        # Step 2: Run face detection and recognition
        faces = app.get(scaled_image)
        
        # Step 3: Compare each detected face's embedding with the user faces
        for face in faces:
            for user_embedding in user_embeddings:
                dist = np.linalg.norm(user_embedding - face.embedding)
                if dist < threshold:
                    shutil.copy(image_path, output_directory)
                    print(f"Copied: {image_path}")
                    return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return False

# Batch processing with multithreading
def process_directory_in_batches(directory_path, user_embeddings, batch_size=100):
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith((".jpg", ".png"))]
    
    # Use ThreadPoolExecutor for multi-threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            futures += [executor.submit(process_image, image_path, user_embeddings) for image_path in batch]

        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Retrieve results to handle exceptions

# Execute the batch processing
process_directory_in_batches(dataset_path, user_embeddings)
