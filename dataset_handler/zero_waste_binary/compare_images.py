from PIL import Image
import numpy as np

def compare_images(image_path1, image_path2):
    """
    Compares two images based on their pixel values.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :return: Mean absolute difference between the images' pixel values.
    """
    # Load the images
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # Check if dimensions are the same
    if img1.size != img2.size:
        raise ValueError("Images must have the same dimensions to compare.")

    # Convert images to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Calculate the absolute difference
    difference = np.abs(img1_array - img2_array)

    # Compute the mean absolute difference
    mean_difference = np.mean(difference)

    return mean_difference

# Example usage
image_path1 = "/home/kaeun.kim/kaeun-dev/android_output4/android_crop/zero_0.00_1.00_0.7713272_1733132657811_crop.webp"
image_path2 = "/home/kaeun.kim/kaeun-dev/android_output4/manual_resize/zero_0.00_1.00_0.7713272_1733132657811.webp"

mean_diff = compare_images(image_path1, image_path2)
print(f"Mean absolute pixel difference: {mean_diff}")
