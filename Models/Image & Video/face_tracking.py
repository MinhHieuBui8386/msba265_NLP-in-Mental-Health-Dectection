import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# Emotion mapping
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loaded model
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_dict)) 
model.load_state_dict(torch.load("resnet34_fer2013_weights30-6300.pth", map_location=device))
model = model.to(device)
model.eval()

# Set image transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Activate webcam or load a video
cap = cv2.VideoCapture('test_video/0001.mp4')  # 0 -> Activate the webcam

# Enable face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize result storage
frame_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Image preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    frame_emotions = []  # Save the detection results for the current frame

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray_frame[y:y+h, x:x+w]
        face = Image.fromarray(face).convert('L')  # Convert to PIL format
        face_tensor = transform(face).unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

        # Predict emotions
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_dict[predicted.item()]

        # Record emotions for the current frame
        frame_emotions.append(emotion)

        # Display the results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save the emotions of the current frame
    frame_results.append(frame_emotions)

    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit webcam (if used)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(frame_results)
df_results.to_csv("data_analysis/test/emotion_analysis_results.csv", index=False)
print("Frame-by-frame emotion analysis results have been saved to emotion_analysis_results.csv")
