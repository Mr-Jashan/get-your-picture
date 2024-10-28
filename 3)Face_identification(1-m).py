import cv2
import os
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize Face Analysis Model
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Load and read the known user's face image
known_image_path = r'training_image\2.jpg'  # Replace with the known user's image
known_image = cv2.imread(known_image_path)

# Detect and get the embedding of the known user's face
known_faces = app.get(known_image)
if len(known_faces) > 0:
    known_embedding = known_faces[0].embedding  # Get the embedding of the first detected face

# Folder where images to be scanned are stored
folder_path = r'dataset'  # Replace with the folder containing images

# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Only process .jpg and .png files
        image_path = os.path.join(folder_path, filename)  # Get full image path
        target_image = cv2.imread(image_path)

        # Detect faces in the target image
        target_faces = app.get(target_image)

        face_found = False
        for face in target_faces:
            target_embedding = face.embedding  # Get the embedding of the current face

            # Compare the known user's embedding with the target face embedding
            similarity = np.dot(known_embedding, target_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(target_embedding))
            print(f"Similarity for {filename}: {similarity}")
            
            # Threshold for determining if faces are a match
            if similarity > 0.45:  # Adjust threshold as needed
                face_found = True
                print(f"User face found in {filename}!")
                box = face.bbox  # Get bounding box of the matched face
                # Draw rectangle around the face
                cv2.rectangle(target_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                # Save or display the image with the matched face
                output_path = os.path.join('output', filename)  # Folder to save output images
                cv2.imwrite(output_path, target_image)  # Save the image with the marked face
                break  # Stop after finding the first match in the current image

        if not face_found:
            print(f"User face not found in {filename}.")

print("Completed scanning all images in the folder.")
