import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

'''
# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''

# ============================
# 1. Data Processing
# ===========================

class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, 0]
        pixels = self.dataframe.iloc[idx, 1]
        
        image = np.array(pixels.split(), dtype='float32').reshape(48, 48)
        image = Image.fromarray(image).convert('L')  # grey to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

# load data
zip_path = 'data_storage/fer2013.zip'
data = pd.read_csv(zip_path, compression='zip')

# split data
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

# Set data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((48, 48)), 
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# create Dataset and DataLoader
train_dataset = FER2013Dataset(train_data, transform=transform)
val_dataset = FER2013Dataset(val_data, transform=transform)
test_dataset = FER2013Dataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================
# 2. bulid ResNet model
# ============================

# load ResNet18,  pre-train model
model = models.resnet34(pretrained=True)

# Modify the output layer to 7 classes (corresponding to FER2013 emotion labels)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# 3. training and validation
# ============================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        # training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward propagation
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

# begining training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
torch.save(model.state_dict(), 'resnet34_fer2013_weights.pth')
print("模型已成功保存！")

# ============================
# 4. test model
# ============================

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

# test model performance
test_model(model, test_loader)