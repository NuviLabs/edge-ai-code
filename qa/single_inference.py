import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

import numpy as np
from typing import List
import heapq
import os

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models, transforms
from glob import glob
import math
import csv

class AIResult:
    def __init__(self, detection_class: int, score: float, rect):
        self.detection_class = detection_class
        self.score = score
        self.rect = rect

    def __repr__(self):
        return f"AIResult(class={self.detection_class}, score={self.score}, rect={self.rect})"

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

class RectF:
    def __init__(self, left: float, top: float, right: float, bottom: float):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def width(self) -> float:
        return int(self.right) - int(self.left)

    def height(self) -> float:
        return int(self.bottom) - int(self.top)

    def __repr__(self):
        return f"RectF(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"


def load_tflite_model(model_path):
    """
    Load a TFLite model from the specified path.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_size):
    """
    Preprocess the input image to fit the YOLO model's expected input size while keeping the aspect ratio.
    Args:
        image_path: Path to the input image.
        input_size: Tuple of (width, height) for resizing.
    Returns:
        Preprocessed image and the original image dimensions.
    """
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size  # (width, height)
    
    # Calculate the new size while keeping the aspect ratio
    # original_width, original_height = original_size
    # target_width, target_height = input_size
    # ratio = min(target_width / original_width, target_height / original_height)
    # new_size = (int(original_width * ratio), int(original_height * ratio))
    # print(new_size)
    
    resized_image = original_image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    
    # Create a new image with the target size and paste the resized image onto it
    # The default pixel value is black (0, 0, 0)
    # img_width, img_height = resized_image.size
    # target_width, target_height = input_size

    # Create a blank (zero-filled) array of the target size with 3 channels (RGB)
    # padded_img_data = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the offsets for centering the image
    # x_offset = (target_width - img_width) // 2
    # y_offset = (target_height - img_height) // 2

    # Convert the resized image to a NumPy array
    resized_image_array = np.array(resized_image)

    # Copy the resized image data into the padded array
    # padded_img_data[
    #     y_offset : y_offset + img_height, x_offset : x_offset + img_width, :
    # ] = resized_image_array

    # # Normalize the padded image to [0, 1] range and add batch dimension
    input_data = np.expand_dims(resized_image_array.astype(np.float32) / 255.0, axis=0)
    # reverse channel
    # input_data = input_data[:, :, :, ::-1]
    return input_data, original_size

def run_inference(interpreter, input_data):
    """
    Run inference on the image using the TFLite interpreter.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    return output_data


def box_iou(rect1, rect2) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        rect1, rect2: RectF objects representing the bounding boxes.
    Returns:
        IoU value as a float.
    """
    # Calculate the intersection area
    inter_left = max(rect1.left, rect2.left)
    inter_top = max(rect1.top, rect2.top)
    inter_right = min(rect1.right, rect2.right)
    inter_bottom = min(rect1.bottom, rect2.bottom)

    if inter_left < inter_right and inter_top < inter_bottom:
        intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    else:
        intersection = 0.0

    # Calculate the union area
    area1 = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
    area2 = (rect2.right - rect2.left) * (rect2.bottom - rect2.top)
    union = area1 + area2 - intersection

    # Handle divide by zero
    return intersection / union if union > 0 else 0.0

def nms(results: List[AIResult]) -> List[AIResult]:
    # same class
    """
    Perform Non-Maximum Suppression (NMS) on a list of detection results.
    Args:
        results: List of AIResult objects containing detection data.
    Returns:
        List of AIResult objects after applying NMS.
    """
    # Priority queue (max heap) based on score
    pq = [(-result.score, result) for result in results]
    heapq.heapify(pq)

    selected_results = []

    while pq:
        _, max_result = heapq.heappop(pq)
        selected_results.append(max_result)

        remaining_results = []
        while pq:
            _, detection = heapq.heappop(pq)
            if box_iou(max_result.rect, detection.rect) < 0.3 and detection.detection_class == max_result.detection_class:  # IoU threshold
                remaining_results.append((-detection.score, detection))

        pq = remaining_results
        heapq.heapify(pq)

    return selected_results

def nms2(results: List[AIResult]) -> List[AIResult]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of detection results.
    Lower priority for classes 7 and 8.
    """
    # Adjust scores for priority classes
    priority_adjusted_results = []
    for result in results:
        adjusted_score = result.score
        # Lower the priority of classes 7 and 8 by reducing their scores
        if result.detection_class in [7, 8]:
            adjusted_score *= 0.5  # Reduce score by half for these classes
        priority_adjusted_results.append((-adjusted_score, result))

    # Create priority queue with adjusted scores
    heapq.heapify(priority_adjusted_results)
    selected_results = []

    while priority_adjusted_results:
        _, max_result = heapq.heappop(priority_adjusted_results)
        selected_results.append(max_result)

        remaining_results = []
        while priority_adjusted_results:
            _, detection = heapq.heappop(priority_adjusted_results)
            if box_iou(max_result.rect, detection.rect) < 0.3:
                # Keep original adjusted score for remaining results
                remaining_results.append((-detection.score * (0.5 if detection.detection_class in [7, 8] else 1.0), detection))

        priority_adjusted_results = remaining_results
        heapq.heapify(priority_adjusted_results)

    return selected_results

def postprocess(outputs, threshold, num_classes, input_size):
    # input_size = (width, height)
    results = []
    # Determine the size of rows and columns from the outputs
    rows = len(outputs[0])
    cols = len(outputs[0][0])

    # Convert outputs to a transposed matrix
    output = np.zeros((cols, rows), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            output[j][i] = outputs[0][i][j]

    for i in range(cols):
        detection_class = -1
        max_score = 0.0
        class_array = output[i][4:4 + num_classes]

        # Find the class with the highest probability
        for j, class_prob in enumerate(class_array):
            if (class_prob > max_score):
                detection_class = j
                max_score = class_prob

        # Determine the score threshold
        score_threshold = threshold

        # Add result if the score exceeds the threshold
        if max_score > score_threshold:
            x_pos = output[i][0]
            y_pos = output[i][1]
            width = output[i][2]
            height = output[i][3]

            # Create a RectF object for the bounding box
            rect_f = RectF(
                int(max(0.0, (x_pos - width / 2.0) * input_size[0])),
                int(max(0.0, (y_pos - height / 2.0) * input_size[1])),
                int(min(input_size[0], (x_pos + width / 2.0) * input_size[0])),
                int(min(input_size[1], (y_pos + height / 2.0) * input_size[1]))
            )

            result = AIResult(detection_class, max_score, rect_f)
            results.append(result)

    filtered_results = nms(results)
    # filtered_results2 = nms2(results)
    return filtered_results


def get_food(output_data, cls_index):
    # filter out rows with confidence scores lower than the threshold
    food_rects = []
    for row in output_data:
        if row.detection_class in cls_index:
            food_rects.append(row.rect)
    return food_rects 

def run_full_inference(detector_path, image_path, save_intermediate=False, save_root=''):
    interpreter = load_tflite_model(detector_path)

    # Get input size
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # Assuming the first input is the image
    input_size = (input_shape[2], input_shape[1])  # Width, Height

    # Preprocess image
    input_data, original_size = preprocess_image(image_path, input_size)

    # Run inference
    output_data = run_inference(interpreter, input_data)

    # Postprocess results
    predictions = postprocess(output_data[0], 0.3, 11, input_size)
    # print(predictions)
    return predictions


def get_tray(detections):
    """
    Get the tray detection result from the list of detections.
    Args:
        detections: List of AIResult objects containing detection data.
        tray_class: Class index for the tray.
        model_config: Model configuration containing input size.
    Returns:
        The best tray detection result or None if no tray is detected.
    """
    tray_detections = [det for det in detections if det.detection_class == 6]

    if not tray_detections:
        print('No tray detected')
        return None

    trays_with_area = []
    for tray in tray_detections:
        area = (tray.rect.right - tray.rect.left) * (tray.rect.bottom - tray.rect.top)
        center_x = (tray.rect.left + tray.rect.right) / 2
        center_y = (tray.rect.top + tray.rect.bottom) / 2

        # Center of the image is (0.5, 0.5)
        distance_from_center = math.sqrt(
            (center_x / 224 - 0.5) ** 2 +
            (center_y / 224 - 0.5) ** 2
        )

        trays_with_area.append({
            'tray': tray,
            'area': area,
            'distance_from_center': distance_from_center
        })

    largest_tray = max(trays_with_area, key=lambda x: x['area'])

    max_allowed_distance = 0.3

    if largest_tray['distance_from_center'] <= max_allowed_distance:
        # print('Using largest tray in center')
        return largest_tray['tray']

    # If largest tray is not in center, use the highest confidence tray
    best_tray = max(tray_detections, key=lambda x: x.score)

    # print(f"Selected tray with confidence: {best_tray.score * 100:.2f}%")

    return best_tray

def get_intersection_area(rect1: RectF, rect2: RectF) -> float:
    """Calculate the intersection area between two rectangles."""
    x_left = max(rect1.left, rect2.left)
    x_right = min(rect1.right, rect2.right)
    y_top = max(rect1.top, rect2.top)
    y_bottom = min(rect1.bottom, rect2.bottom)
    
    if x_right > x_left and y_bottom > y_top:
        return (x_right - x_left) * (y_bottom - y_top)
    return 0.0

def get_food_area_inside_tray(tray: AIResult, results: List[AIResult]) -> float:
    """
    Calculate the total area of food inside the tray, excluding overlapping areas.
    Special handling for class 5 objects in bottom right region.
    """
    food_regions = []
    
    for det in results:
        if det.detection_class == 0 or det.detection_class == 5:
            center_x = (det.rect.left + det.rect.right) / 2
            center_y = (det.rect.top + det.rect.bottom) / 2

            # For class 5, check if it's in the bottom right region
            if det.detection_class == 5:
                # Define bottom right region (last 1/3 of width and height)
                tray_right_bound = tray.rect.left + (tray.rect.right - tray.rect.left) * 1/2
                tray_bottom_bound = tray.rect.top + (tray.rect.bottom - tray.rect.top) * 1/2
                
                # Skip if not in bottom right region
                if center_x < tray_right_bound or center_y < tray_bottom_bound:
                    continue

            if (tray.rect.left <= center_x <= tray.rect.right and
                tray.rect.top <= center_y <= tray.rect.bottom):
                # Get the intersection with tray
                intersection_left = max(tray.rect.left, det.rect.left)
                intersection_right = min(tray.rect.right, det.rect.right)
                intersection_top = max(tray.rect.top, det.rect.top)
                intersection_bottom = min(tray.rect.bottom, det.rect.bottom)
                
                food_rect = RectF(intersection_left, intersection_top, 
                                intersection_right, intersection_bottom)
                food_regions.append(food_rect)
    
    total_area = 0.0
    processed_regions = []
    
    # Process food regions and subtract overlapping areas
    for region in food_regions:
        current_area = region.width() * region.height()
        overlap_area = 0.0
        
        # Calculate overlaps with previously processed regions
        for processed in processed_regions:
            overlap_area += get_intersection_area(region, processed)
            
        # Add the non-overlapping area to total
        total_area += current_area - overlap_area
        processed_regions.append(region)
    
    return total_area

def run_entire_pipeline(model_path, webp):
    detection_results = run_full_inference(model_path, webp, save_intermediate=False, save_root='/mnt/nas/data/kaeun/q4/binary_zerowaste/food_binary_original/zero_waste')
    print('Detection results:')
    for det in detection_results:
        print(det)
    
    label_exists = {det.detection_class: False for det in detection_results}
    for det in detection_results:
        label_exists[det.detection_class] = True


    tray = get_tray(detection_results)
    if tray is None:
        return -1, -1, label_exists, -1, -1
    food_in_tray = get_food_area_inside_tray(tray, detection_results)
    tray_area = (tray.rect.right - tray.rect.left) * (tray.rect.bottom - tray.rect.top)
    zero_waste_score = food_in_tray / tray_area
    return_zero_waste_score = float(zero_waste_score)
    if zero_waste_score >= 0.3:
        zero_waste_score = 0.3
    intake_score = 1 - zero_waste_score/0.3
    print("food_in_tray: ", food_in_tray)
    print("tray_area: ", tray_area)
    print("zero_waste_score: ", zero_waste_score)
    return intake_score, return_zero_waste_score, label_exists, tray_area, food_in_tray

if __name__ == "__main__":
    # Paths
    model_path = "/home/kaeun.kim/ultralytics/runs/segment/train/weights/best_saved_model/best_float16.tflite"  # Replace with your YOLO TFLite model path
    img_path = "/home/kaeun.kim/kaeun-dev/nuvilab/qa/kids_detector/prugioai-dc_241220_121110_27823_L_A_DD-00000649_Trayfile.png"

    run_entire_pipeline(model_path, img_path)
