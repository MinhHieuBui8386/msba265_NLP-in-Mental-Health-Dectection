import cv2 
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

# Emotion mapping
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# loaded model 
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_dict)) 
model.load_state_dict(torch.load("CNN/resnet34_fer2013_weights30-6300.pth", map_location=device))
model = model.to(device)
model.eval()

# Set image transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
        transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Activate the webcam or load a video
cap = cv2.VideoCapture(0) 

# Enable face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Image preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray_frame[y:y+h, x:x+w]
        face = Image.fromarray(face).convert('L')  # 轉換為 PIL 格式
        face_tensor = transform(face).unsqueeze(0).to(device)  # 添加 batch 維度並移至 GPU/CPU

        # Predict emotion
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_dict[predicted.item()]

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
