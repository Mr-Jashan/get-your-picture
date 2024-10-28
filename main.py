import cv2
import os
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize Face Analysis Model
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Folder containing multiple images of the known user with different appearances
known_user_folder = r'training_image/'  # Replace with folder containing alternate appearances

# Load and store embeddings for all known user images
known_embeddings = []
for filename in os.listdir(known_user_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        known_image_path = os.path.join(known_user_folder, filename)
        known_image = cv2.imread(known_image_path)
        
        # Detect and get embedding for each known user image
        known_faces = app.get(known_image)
        if len(known_faces) > 0:
            known_embeddings.append(known_faces[0].embedding)

# Folder where images to be scanned are stored
folder_path = r'dataset/'  # Replace with the folder containing target images

# Loop through each image in the target folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        target_image = cv2.imread(image_path)

        # Detect faces in the target image
        target_faces = app.get(target_image)

        face_found = False
        for face in target_faces:
            target_embedding = face.embedding  # Get embedding of the current face

            # Compare against all known user embeddings
            for known_embedding in known_embeddings:
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
                    output_path = os.path.join('output', filename)
                    cv2.imwrite(output_path, target_image)  # Save the image with the marked face
                    break  # Stop after finding the first match in the current image
            if face_found:
                break

        if not face_found:
            print(f"User face not found in {filename}.")

print("Completed scanning all images in the folder.")
