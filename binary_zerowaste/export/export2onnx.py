import torch
from torchvision import models, transforms
from PIL import Image
from glob import glob
import torch.nn as nn
# import ai_edge_torch
# import torch

model_name = "best_model_yolo_m_tflite4"
# Define the model architecture
model = models.resnet18(weights='IMAGENET1K_V1')  # Match your training setup
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary classification
# Load the trained weights
model.load_state_dict(torch.load(f"/home/kaeun.kim/kaeun-dev/nuvilab/models/{model_name}.pth", map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()
# Add a static Reshape layer for TensorFlow compatibility

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)  # Match input shape
onnx_path = f"{model_name}.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes=None
)
print(f"ONNX model with static reshape saved at {onnx_path}")
