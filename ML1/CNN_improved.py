import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Use GradScaler for mixed precision training
scaler = GradScaler()  # Updated to the new syntax

# Safe loader function to handle image loading errors
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Use GradScaler for mixed precision training
scaler = GradScaler()  # Updated to the new syntax

# Safe loader function to handle image loading errors
def safe_loader(path):
    try:
        img = Image.open(path)
        img.verify()  # Verify if the image is valid
        img = Image.open(path)  # Re-open the image after verification
        return img
    except (IOError, SyntaxError) as e:
        print(f"Error loading image {path}: {e}")
        return None

def train():
    # Step 1: Load and Split Dataset
    data_dir = './generate_data'

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),              # Resize images to 128x128
        transforms.RandomHorizontalFlip(),          # Random horizontal flip for augmentation
        transforms.RandomRotation(30),              # Random rotation up to 30 degrees
        transforms.ToTensor(),                      # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Load the dataset using ImageFolder with the safe loader
    dataset = datasets.ImageFolder(data_dir, transform=transform, loader=safe_loader)

    # Filter out any invalid images that returned None
    dataset.samples = [sample for sample in dataset.samples if sample[0] is not None]

    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define DataLoader for training and validation datasets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Step 2: Define Model Architecture
    class CNNModel(nn.Module):
        def __init__(self, num_classes):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)  # MaxPooling with a 2x2 kernel
            self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Fully connected layer
            self.fc2 = nn.Linear(512, num_classes)  # Output layer

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, 128 * 16 * 16)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Step 3: Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and move it to the device (GPU if available)
    num_classes = len(dataset.classes)
    model = CNNModel(num_classes).to(device)  # Move model to GPU

    # Step 4: Set Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 5: Training Loop with Mixed Precision
    num_epochs = 10  # Number of epochs to train the model

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over the training data
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # Move inputs and labels to the device (GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero out the gradients from the previous step

            # Mixed precision: using autocast for automatic casting to FP16
            with autocast(device_type=device.type):  # Specify the device type (CPU or CUDA)
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Calculate the average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)

        # Step 6: Validation Loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate the average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        # Step 7: Output Training Results
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy*100:.2f}%")

    # Step 8: Save the Model
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved to cnn_model.pth")

if __name__ == '__main__':
    train()
