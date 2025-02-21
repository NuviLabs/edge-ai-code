import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

experiment_name = "yolo_m_tflite4"

# Load pretrained ResNet-18
model = models.resnet18(pretrained=True)

# Modify the fully connected layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 1)  # Output layer with 1 neuron

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define data augmentations for training and validation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0)),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ImageFolder(root='/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_cropped_tflite/train', transform=train_transforms)
val_dataset = ImageFolder(root='/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_cropped_tflite/val', transform=val_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Training and validation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


best_val_loss = float('inf')
epoch = 500

for epoch in range(epoch):  # Example: 10 epochs
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Calculate average validation loss
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model_{experiment_name}.pth")
        print(f"New best model saved with Validation Loss: {best_val_loss}")

    
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

# Save the model weights after training
# torch.save(model.state_dict(), "resnet18_binary_classification.pth")
# print("Model weights saved to 'resnet18_binary_classification.pth'")
