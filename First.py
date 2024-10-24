import cv2
import insightface
from insightface.app import FaceAnalysis

# Initialize Face Analysis Model
app = FaceAnalysis()  # Loads the default model
app.prepare(ctx_id=0, det_size=(640, 640))  # Prepares the model with context and detection size

# Load and read the image where you want to perform face recognition
image_path = r'dataset\4.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Detect faces in the image
faces = app.get(image)

# Draw bounding boxes and show the image with detected faces
for face in faces:
    box = face.bbox  # Get bounding box of the face
    # Draw rectangle around the face
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
    # Print confidence and facial landmarks
    print(f"Confidence: {face.det_score}")
    print(f"Facial landmarks: {face.landmark}")

# Show the image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
