from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models, transforms
from glob import glob

# Load the PyTorch model
model = models.resnet18(weights='IMAGENET1K_V1')  # Match your training setup
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary classification
model.load_state_dict(torch.load("/home/kaeun.kim/kaeun-dev/nuvilab/models/best_model_yolo_m_tflite.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_path, input_size=(224, 224)):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Define transformations (ensure these match your training settings)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# Load and preprocess the image
image_tensor = preprocess_image("/home/kaeun.kim/kaeun-dev/android_output4/android_crop/zero_0.00_1.00_0.53285056_1733129152043_crop.webp")

# Perform inference
with torch.no_grad():
    pytorch_output = model(image_tensor).squeeze().item()
    print(f"PyTorch Model Output: {pytorch_output}")

# Convert PyTorch tensor to NumPy array
# image tensor [1, 3, 224, 224]
tflite_input = image_tensor.numpy()

import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/kaeun.kim/kaeun-dev/nuvilab/models/export/best_model_yolo_m_tflite.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], tflite_input)

# Run inference
interpreter.invoke()

# Get the output tensor
tflite_output = interpreter.get_tensor(output_details[0]['index']).squeeze()
print(f"TFLite Model Output: {tflite_output}")
