import os
import cv2
from insightface.app import FaceAnalysis
import numpy as np

# Initialize the InsightFace app for CPU usage
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU (ctx_id=-1)

# Function to extract the face embedding from an image
def get_face_embedding_traning(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Detect faces and extract embeddings
    faces = app.get(img)
    if faces:
        return faces[0].embedding  # Return the embedding of the first detected face
    else:
        print(f"No face detected in image: {image_path}")
        return None

def get_face_embeddings_test(image_path):
    
    # Load image
    image = cv2.imread(image_path)
    
    # Detect faces in the image and extract embeddings
    faces = app.get(image)
    
    embeddings = []
    
    for face in faces:
        # Each face has an embedding attribute
        embeddings.append(face.embedding)
    
    return embeddings

# Function to compare embeddings and check if they match
def is_face_match(embedding1, embedding2, threshold=6):
    dist = np.linalg.norm(embedding1 - embedding2)
    print(dist)
    return dist < threshold  # Returns True if the distance is less than the threshold

# Function to compute the average embedding from multiple images
def get_average_embedding(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Consider only image files
            image_path = os.path.join(folder_path, filename)
            embedding = get_face_embedding_traning(image_path)
            if embedding is not None:
                embeddings.append(embedding)

    if embeddings:
        # Compute the average of all embeddings
        average_embedding = np.mean(embeddings, axis=0)
        return average_embedding
    else:
        print("No valid face embeddings found in the training folder.")
        return None

# Path to the folder containing multiple training images
training_folder_path = r"training_image"

# Load the average face embedding for the training images
training_embedding = get_average_embedding(training_folder_path)

if training_embedding is None:
    print("No face embeddings found in the training images. Exiting.")
    exit()

# Path to the test image (image to check against the trained face)
test_image_path = r"dataset\16.png"  # Adjust as needed

# Get the embedding for the test image
test_embedding = get_face_embeddings_test(test_image_path)

if test_embedding is None:
    print("No face found in the test image.")
else:
    # Check if the test image matches the average training embedding
    if is_face_match(training_embedding, test_embedding):
        print("Yes, the face matches!")
    else:
        print("No, the face does not match.")
