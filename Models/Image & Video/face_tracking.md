# Emotion mapping
```
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
```
- This code defines the emotion labels for a facial emotion recognition task.
- A dictionary is created to map numerical labels (from model predictions) to actual emotion names.
# Set device
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
- The code checks if a GPU is available and selects it for model inference if possible. If not, it defaults to using the CPU.
# Loaded model
```
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_dict)) 
model.load_state_dict(torch.load("resnet34_fer2013_weights30-6300.pth", map_location=device))
model = model.to(device)
model.eval()
```
- This code snippet is focused on setting up a ResNet-34 model for facial emotion recognition by modifying the output layer and loading pre-trained weights.
- Prepares the model for running predictions on the emotion dataset, enabling it to classify emotions in images based on the trained weights.
# Activate webcam or load a video
```
cap = cv2.VideoCapture('test_video/0001.mp4')  # 0 -> Activate the webcam
```
- This code snippet is used for initializing the video capture source in OpenCV.
- Prepares the system to capture video, either from a file ('test_video/0001.mp4') or from a webcam (if 0 is used). The cap object will be used to read frames from this video source in further steps.
# Enable face detection
```
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
- This code snippet sets up face detection using OpenCV's Haar Cascade Classifier. 
- Initializes the face_detector using OpenCV's pre-trained Haar cascade classifier for detecting faces in images or video. The face_detector object can then be used to detect faces in subsequent frames of a video or image.
# Initialize result storage
```
frame_results = []
```
- This list is used to store the results of each processed frame. Each element of the list will likely contain data about detected faces and the corresponding predicted emotions for a specific frame in a video stream.
```
while True:
    ret, frame = cap.read()
    if not ret:
        break
```
- Main Processing Loop
# Image preprocessing
```
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
```
- Converts the captured frame from color (BGR) to grayscale. Grayscale images are required for face detection using the Haar Cascade classifier.
- Detects faces in the grayscale frame using the previously initialized face detector
# Saving Detection Results
```
frame_emotions = []
```
- Save the detection results for the current frame
# Extracting the Face Region
```
for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = Image.fromarray(face).convert('L')  
        face_tensor = transform(face).unsqueeze(0).to(device)  
```
- This loop iterates over each detected face in the frame.
- Crops the detected face region from the grayscale frame using the coordinates and dimensions.
- Converts the cropped face image into a PIL format and changes the color mode to grayscale.
# Predicting Emotions
```
with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_dict[predicted.item()]
```
- Passes the face tensor through the model to get the emotion prediction scores.
- Maps the predicted class index to the corresponding emotion label from the emotion_dict.
# Record emotions for the current frame
```
frame_emotions.append(emotion)
```
- Adds the predicted emotion for the current face to the frame_emotions list. This allows storing emotions for all faces detected in the current frame.
# Display the results
```
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
```
- Draws a rectangle around the detected face using the coordinates (x, y) for the top-left corner and (x+w, y+h) for the bottom-right corner. The rectangle is drawn in green with a thickness of 2.
- Displays the predicted emotion near the detected face in red text.
# Save the emotions of the current frame
```
frame_results.append(frame_emotions)
cv2.imshow("Emotion Detection", frame)
```
- Adds the list of emotions for the current frame to frame_results, which stores emotions for all frames processed in the video.
- Displays the current video frame with the emotion detection results, including the face rectangle and the emotion text.
# Press 'q' to exit webcam (if used)
```
if cv2.waitKey(1) & 0xFF == ord('q'):
break
```
- If the 'q' key is pressed, the loop breaks, stopping the video capture and emotion detection process.
# Releasing the Video Capture and Closing Windows
```
cap.release()
cv2.destroyAllWindows()
```
# Convert results to DataFrame and save as CSV
```
df_results = pd.DataFrame(frame_results)
df_results.to_csv("data_analysis/test/emotion_analysis_results.csv", index=False)
print("Frame-by-frame emotion analysis results have been saved to emotion_analysis_results.csv")
```
- The results of the emotion analysis (stored in frame_results) are converted into a pandas DataFrame.
- The results are saved as a CSV file named emotion_analysis_results.csv in the specified directory.
- A success message is printed to confirm that the process is complete.
