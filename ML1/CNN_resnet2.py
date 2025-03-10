import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (if input and output channels don't match, use 1x1 convolution)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add the shortcut (residual connection)
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create the layers of residual blocks
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(512, num_classes)  # Fully connected output layer
        self.dropout = nn.Dropout(0.5)  # Add Dropout layer to prevent overfitting

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)
        return x

def train():
    # Step 1: Load and Split Dataset
    data_dir = './generate_data1'

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),              # Resize images to 128x128
        transforms.RandomHorizontalFlip(),          # Random horizontal flip for augmentation
        transforms.RandomRotation(30),              # Random rotation up to 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    model = ResNetModel(num_classes).to(device)

    # Step 3: Set Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Use AdamW optimizer for better regularization
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler

    # Step 4: Training Loop with Mixed Precision
    num_epochs = 20  # Number of epochs to train the model

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

        # Step 5: Validation Loop
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

        # Step 6: Output Training Results
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy*100:.2f}%")

        scheduler.step()  # Update learning rate

    # Step 7: Save the Model
    torch.save(model.state_dict(), "resnet_model2.pth")
    print("Model saved to resnet_model2.pth")

if __name__ == '__main__':
    train()
