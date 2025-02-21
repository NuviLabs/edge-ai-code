import os
import cv2
from PIL import Image
import numpy as np
from run_tflite import run_full_inference
from datetime import datetime

def rescale_coordinates(rect, original_size, model_size=(224, 224)):
    """
    Rescale coordinates from model size to original image size.
    
    Args:
        rect: RectF object with coordinates
        original_size: Tuple of (width, height) of original image
        model_size: Tuple of (width, height) of model input size
    Returns:
        RectF object with rescaled coordinates
    """
    width_ratio = original_size[0] / model_size[0]
    height_ratio = original_size[1] / model_size[1]
    
    rect.left = int(rect.left * width_ratio)
    rect.right = int(rect.right * width_ratio)
    rect.top = int(rect.top * height_ratio)
    rect.bottom = int(rect.bottom * height_ratio)
    
    return rect

def draw_detections(image_path, detections, output_path, class_colors=None):
    """
    Draw detection results on the image and save it.
    """
    # Default colors if none provided (BGR format)
    if class_colors is None:
        class_colors = {
            0: (0, 255, 0),    # Food (Green)
            6: (255, 0, 0),    # Tray (Blue)
            4: (0, 165, 255),  # Rice (Orange)
            9: (0, 0, 255),    # Sauce (Red)
        }

    # Read image and get original size
    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Draw each detection
    for det in detections:
        # Rescale coordinates to original image size
        scaled_rect = rescale_coordinates(det.rect, original_size)
        
        # Get coordinates
        x1 = int(scaled_rect.left)
        y1 = int(scaled_rect.top)
        x2 = int(scaled_rect.right)
        y2 = int(scaled_rect.bottom)
        
        # Get color for class (default to white if class not in colors dict)
        color = class_colors.get(det.detection_class, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with class and confidence
        label = f"Class {det.detection_class}: {det.score:.2f}"
        # Scale font size based on image size
        font_scale = min(original_size) / 1000.0
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, 2)

    # Save the annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def main():
    # Paths
    model_path = "/home/kaeun.kim/ultralytics/runs/segment/dupbab_remove_sauce_filter_hm/weights/best_saved_model/best_float16.tflite"
    input_dir = "/mnt/nas/data/kaeun/2025q1/kids/testset/images"  # Replace with your input directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/mnt/nas/data/kaeun/2025q1/kids/testset/test_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images in input directory
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png', '.webp')) and 'difficult' not in img_file:
            if 'moarae-dc_241224_122308_30178_L_A_DD-00000733_Trayfile' not in img_file:
                continue
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"{img_file}")
            
            # Run detection
            detections = run_full_inference(model_path, img_path, num_cls=11)
            
            # Draw and save visualizations
            draw_detections(img_path, detections, output_path)
            
            # Print results
            print(f"\nProcessed {img_file}")
            print(f"Number of detections: {len(detections)}")

if __name__ == "__main__":
    main()
