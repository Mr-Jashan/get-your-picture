import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize Face Analysis Model
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Load and read the known user's face image
known_image_path = r'training_image\3.jpg'  # Replace with your known user image
known_image = cv2.imread(known_image_path)

# Detect and get the embedding of the known user's face
known_face = app.get(known_image)
if len(known_face) > 0:
    known_embedding = known_face[0].embedding  # Get the embedding of the first detected face

# Load and read the target image where we want to check for the user's face
target_image_path = r'dataset/6.png'  # Replace with the target image path
target_image = cv2.imread(target_image_path)

# Detect faces in the target image
target_faces = app.get(target_image)

# Loop over each detected face in the target image
face_found = False
for face in target_faces:
    target_embedding = face.embedding  # Get the embedding of the current face

    # Compare the known user's embedding with the target face embedding
    similarity = np.dot(known_embedding, target_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(target_embedding))
    print(similarity)
    # Threshold for determining if faces are a match (adjustable, around 0.6 to 0.7 is usually good)
    if similarity > 0.45:  # 1 means identical, so 0.6 or above is considered a match
        face_found = True
        print("User face found!")
        box = face.bbox  # Get bounding box of the matched face
        # Draw rectangle around the face
        cv2.rectangle(target_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        break  # Stop after finding the first match

if not face_found:
    print("User face not found in the image.")

# Show the image with detected faces (if any)
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 400, 300) 
cv2.imshow('Result', target_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
