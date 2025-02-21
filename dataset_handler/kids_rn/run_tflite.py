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

class AIResult:
    def __init__(self, detection_class: int, score: float, rect):
        self.detection_class = detection_class
        self.score = score
        self.rect = rect

    def __repr__(self):
        return f"AIResult(class={self.detection_class}, score={self.score}, rect={self.rect})"

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
    original_width, original_height = original_size
    target_width, target_height = input_size
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    
    resized_image = original_image.resize(new_size, resample=Image.Resampling.BILINEAR)
    
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", input_size)
    new_image.paste(resized_image, ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2))
    
    input_data = np.expand_dims(np.array(new_image, dtype=np.float32) / 255.0, axis=0)
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
    """
    Perform Non-Maximum Suppression (NMS) on a list of detection results.
    Args:
        results: List of AIResult objects containing detection data.
    Returns:
        List of AIResult objects after applying NMS.
    """
    # Priority queue (max heap) based on score
    pq = [(-result.score, result) for result in results]
    # pq = [(-result.score, result) for result in results if result.detection_class != 8]
    heapq.heapify(pq)

    selected_results = []

    while pq:
        _, max_result = heapq.heappop(pq)
        selected_results.append(max_result)

        remaining_results = []
        while pq:
            _, detection = heapq.heappop(pq)
            if box_iou(max_result.rect, detection.rect) < 0.7:# and detection.detection_class == max_result.detection_class:  # IoU threshold
                remaining_results.append((-detection.score, detection))

        pq = remaining_results
        heapq.heapify(pq)

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
            if class_prob > max_score:
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

    return nms(results[:300])


def get_food(output_data, cls_index):
    # filter out rows with confidence scores lower than the threshold
    food_rects = []
    for row in output_data:
        if row.detection_class in cls_index:
            food_rects.append(row.rect)
    return food_rects 



def draw_bounding_boxes(image_path, predictions, save_path, input_size=(224, 224)):
    """
    Draw bounding boxes on the original image and save the result.
    Args:
        image_path: Path to the original image.
        predictions: List of predictions with (box, score, class_id).
        save_path: Path to save the image with bounding boxes.
        original_size: Original size of the input image.
        input_size: Size of the input image to the model.
    """
    image = cv2.imread(image_path)
    orig_width, orig_height = image.shape[1], image.shape[0] 
    for pred in predictions:
        x1 = int(pred.rect.left * orig_width / input_size[0])
        y1 = int(pred.rect.top * orig_height / input_size[1])
        x2 = int(pred.rect.right * orig_width / input_size[0])
        y2 = int(pred.rect.bottom * orig_height / input_size[1])
        score = pred.score
        class_id = pred.detection_class
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(save_path, image)

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
    predictions = postprocess(output_data[0], 0.25, 9, input_size)
    # print predictions in a clear way
    for pred in predictions:
        print(pred)
    
    # Draw bounding boxes and save the image
    if save_intermediate:
        save_path = os.path.join(save_root, os.path.basename(image_path))
        draw_bounding_boxes(image_path, predictions, save_path, original_size, input_size)

if __name__ == "__main__":
    # Paths
    model_path = "/home/kaeun.kim/yolov11/runs/segment/train4/weights/best_saved_model/best_float16.tflite"  # Replace with your YOLO TFLite model path

    # image paths
    from glob import glob
    # webps = glob("/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/train/zero_waste/*.jpeg")
    webps = ["/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_rn/dotori-sopoong-nexonhae-dc_241220_125011_27523_L_A_DD-00000711_Trayfile.png"]
    preds = []
    for webp in webps:
        if 'crop' not in webp:
            print(f"==== Processing {webp} ===== ")
            run_full_inference(model_path, webp, save_intermediate=False, save_root='/mnt/nas/data/kaeun/q4/binary_zerowaste/food_binary_original/zero_waste')
            # draw_bounding_boxes(webp, preds, webp.replace('.png', '_crop.png'))

    print('')

    # dataset_path = '/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/train/zero_waste'
    # images = glob(dataset_path + '/*.jpeg')


