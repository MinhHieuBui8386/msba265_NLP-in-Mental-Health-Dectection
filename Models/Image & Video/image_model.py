import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, 0]
        pixels = self.dataframe.iloc[idx, 1]

        # Image processing
        image = np.array(pixels.split(), dtype='float32').reshape(48, 48)
        image = Image.fromarray(image).convert('L')  # Convert grayscale to RGB

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Combine FER2013 and Facial Emotion data
def load_and_combine_data(fer2013_path, additional_data_path, dataset_dir):
    # loaded fer2013 dataste
    fer_data = pd.read_csv(fer2013_path, compression='zip')
    fer_data = fer_data[~fer_data['emotion'].isin([1, 2])]  # Remove "disgust" and "fear"
    fer_data.reset_index(drop=True, inplace=True)

    # Adjust label encoding
    def adjust_label(x):
        if x > 2:
            return x - 2  # Shift labels 3-6 forward by 2
        return x  # Keep labels 0 and 1 unchanged
    
    fer_data['emotion'] = fer_data['emotion'].apply(adjust_label)

    # Load additional data
    additional_data = pd.read_csv(additional_data_path)
    images, labels = [], []

    for _, row in additional_data.iterrows():
        img_path = os.path.join(dataset_dir, row['filename'])
        img = Image.open(img_path).convert('L').resize((48, 48))
        images.append(' '.join(map(str, np.array(img).flatten())))
        labels.append(row['label'])

    additional_data_df = pd.DataFrame({'emotion': labels, 'pixels': images})

    # Combine data
    combined_data = pd.concat([fer_data, additional_data_df], ignore_index=True)
    return combined_data

# Load data
zip_path = '../data_storage/fer2013.zip'
additional_csv = '../data_storage/facial_adjusted.csv'
dataset_dir = '../data_storage/dataset'

combined_data = load_and_combine_data(zip_path, additional_csv, dataset_dir)
combined_data_clean = combined_data.drop(columns=['Usage'])

label_to_emotion = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Neutral"
}
combined_data_dist = combined_data_clean['emotion'].map(label_to_emotion).value_counts()


def split_data(combined_data_clean):
    from sklearn.model_selection import train_test_split

    # Split training/validation set and test set
    train_val_data, test_data = train_test_split(
        combined_data_clean, 
        test_size=0.2, 
        random_state=42, 
        stratify=combined_data['emotion']
    )

    # Split training set and validation set
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_val_data['emotion']
    )

    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(combined_data_clean)

# Define data augmentation
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = EmotionDataset(train_data, transform=transform)
val_dataset = EmotionDataset(val_data, transform=transform)
test_dataset = EmotionDataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# load ResNet,  pre-train model
model = models.resnet50(pretrained=True)

# Modify the output layer to 5 classes
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
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
        
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved successfully!")

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
    
test_model(model, test_loader)