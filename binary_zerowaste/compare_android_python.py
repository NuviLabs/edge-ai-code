import torch
from torchvision import models, transforms
from PIL import Image
from glob import glob

# Define the model architecture
model = models.resnet18(weights='IMAGENET1K_V1')  # Match your training setup
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary classification

# Load the trained weights
model_path = "/home/kaeun.kim/kaeun-dev/nuvilab/models/best_model_yolo_m_tflite.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Define the same transforms used during training/validation
inference_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # Ensure consistent input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image

android_imgs = glob('/home/kaeun.kim/kaeun-dev/android_output2/android_crop2/*.webp')
manually_cropped_imgs = glob('/home/kaeun.kim/kaeun-dev/android_output2/manual_crop2/*.webp')
device_output = []
android_outputs = []
manual_outputs = []
for ai, mi in zip(android_imgs, manually_cropped_imgs):
    # get device_output value
    if float(ai.split('/')[-1].split('_')[3]) >= 0.5:
        device_output.append(1)
    else:
        device_output.append(0)
    #android
    image = Image.open(ai).convert("RGB")  # Ensure the image is in RGB format
    input_tensor = inference_transforms(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert logits to probabilities
    probability = torch.sigmoid(output.squeeze()).item()

    # Convert probability to binary class (0 or 1)
    threshold = 0.5  # You can adjust the threshold based on your needs
    prediction_ai = 1 if probability >= threshold else 0


    # manually cropped
    image = Image.open(mi).convert("RGB")  # Ensure the image is in RGB format
    input_tensor = inference_transforms(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert logits to probabilities
    probability = torch.sigmoid(output.squeeze()).item()

    # Convert probability to binary class (0 or 1)
    threshold = 0.5  # You can adjust the threshold based on your needs
    prediction_mi = 1 if probability >= threshold else 0

    android_outputs.append(prediction_ai)
    manual_outputs.append(prediction_mi)

print('')
