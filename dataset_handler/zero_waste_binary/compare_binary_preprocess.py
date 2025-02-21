from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

# Custom implementation (from previous step)
def custom_transform(image_path, input_size):
    # Load the image
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels

    # Step 1: Resize
    resized_image = image.resize(input_size, resample=Image.Resampling.BILINEAR)

    # Step 2: Convert to NumPy array and scale to [0, 1]
    np_image = np.array(resized_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # Step 3: Normalize using mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Step 4: Rearrange dimensions to [C, H, W] for PyTorch (Channel First)
    np_image = np.transpose(np_image, (2, 0, 1))  # HWC -> CHW

    # Step 5: Convert to PyTorch tensor
    tensor_image = torch.tensor(np_image, dtype=torch.float32)

    return tensor_image

# PyTorch transforms pipeline
def transforms_pipeline(image_path, input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Compare the results
def compare_transforms(image_path, input_size):
    # Outputs from both methods
    custom_output = custom_transform(image_path, input_size)
    transforms_output = transforms_pipeline(image_path, input_size)

    # Check if they are close
    are_equal = torch.allclose(custom_output, transforms_output, atol=1e-6)

    print("Custom Transform Output Shape:", custom_output.shape)
    print("Transforms Output Shape:", transforms_output.shape)
    print("Are the outputs identical?:", are_equal)

    if not are_equal:
        # Show difference if not identical
        difference = torch.abs(custom_output - transforms_output)
        print("Maximum Difference:", torch.max(difference))

# Example usage
image_path = "/home/kaeun.kim/kaeun-dev/android_output4/manual_crop/zero_0.00_0.00_0.003707253_1733123592518.webp"  # Replace with your image path
input_size = (224, 224)
compare_transforms(image_path, input_size)
