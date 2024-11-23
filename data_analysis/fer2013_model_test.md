```Python
import cv2 
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
```
 Import OpenCV, a library for computer vision tasks. It is used here for face detection and video processing.<br>
 Import NumPy, a library for numerical computations. It is used here to handle array operations for image processing.<br>
 Import PyTorch, a deep learning framework. It is used here for model loading, inference, and handling tensors.<br>
 Import `transforms` and `models` from torchvision.<br>
 - `transforms`: For preprocessing images before feeding them into the model.<br>
 - `models`: For loading pre-trained or custom models.<br>
 Import the Python Imaging Library (Pillow) to handle image conversions.<br>
 It is used here to convert NumPy arrays (from OpenCV) into image objects compatible with PyTorch.<br>


```Python
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
```
 This is a dictionary that maps numeric labels (0 to 6) to their corresponding emotion names.<br>
 The numeric labels are the output from the deep learning model (classification indices).<br>
 - Key (integer): The label predicted by the model.<br>
 - Value (string): The human-readable emotion corresponding to that label.<br>
 - If the model outputs 3, the predicted emotion is "Happy".<br>
 - If the model outputs 5, the predicted emotion is "Sad".<br>


 loaded model 
```Python
 loaded model 
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_dict)) 
model.load_state_dict(torch.load("CNN/resnet34_fer2013_weights30-6300.pth", map_location=device))
odel = model.to(device)
model.eval()
```
 Initializes a ResNet34 model from `torchvision.models`.
 - `pretrained=False`: Indicates that the model will not load pre-trained weights on ImageNet. 
 Modifies the fully connected (fc) layer of ResNet34.
 - `model.fc.in_features`: Retrieves the number of input features to the last layer of ResNet34 (usually 512).
 - `len(emotion_dict)`: Sets the number of output classes to match the number of emotions (7 in this case).
 This customization ensures the model outputs predictions for our emotion recognition task.
 Loads the pre-trained weights for the custom ResNet34 model.
 - `torch.load("CNN/resnet34_fer2013_weights30-6300.pth")`: Reads the saved weights file from the specified path.
 - `map_location=device`: Ensures the weights are loaded onto the correct device (CPU or GPU) dynamically.
 Moves the entire model to the specified device (GPU or CPU), enabling it to perform computations there.
 Sets the model to evaluation mode.
 - In evaluation mode, certain layers like dropout or batch normalization behave differently (e.g., no random dropout is applied).
 This is essential to ensure consistent predictions during inference.


```python
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_dict))
model.load_state_dict(torch.load("CNN/resnet34_fer2013_weights30-6300.pth", map_location=device))
model = model.to(device)
model.eval()
```
 - This code snippet is designed to load a pre-trained model (specifically ResNet-34) and adapt it to the facial emotion recognition task. It includes steps for modifying the model's output layer and loading pre-trained weights.
 - The ResNet-34 model is loaded with a modified final layer to match the number of emotion categories in the dataset.
 - Pre-trained weights from a saved checkpoint (resnet34_fer2013_weights.pth) are loaded to initialize the model.
 - The model is then moved to the appropriate device (GPU or CPU) and set to evaluation mode for inference.

 ```python
 transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```
- This code snippet sets up a series of image transformations that are applied to the images in the dataset before they are fed into a model.
- Resize: Ensures that all images are the same size (48x48), which is required for the neural network.
- Grayscale: Converts images to grayscale with three channels, standardizing the input format for the model.
- ToTensor: Converts the image to a tensor, which is needed for processing in PyTorch.
- Normalize: Centers the image pixel values, helping the model learn more efficiently.

```python
cap = cv2.VideoCapture(0)
```
- This line of code sets up the video capture using either a camera or a video file.

```python
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        
        face = gray_frame[y:y+h, x:x+w]
        face = Image.fromarray(face).convert('L')  
        face_tensor = transform(face).unsqueeze(0).to(device)  

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_dict[predicted.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
- This code snippet is a real-time emotion detection program using OpenCV and PyTorch. It performs face detection and emotion classification using a pre-trained model.
- Captures video frames from the webcam.
- Pre-processes each detected face, transforms it, and makes emotion predictions using a pre-trained model.
- Draws rectangles around detected faces and displays the predicted emotion on the screen.



