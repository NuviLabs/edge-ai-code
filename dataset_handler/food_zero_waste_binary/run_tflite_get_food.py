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
    Preprocess the input image to fit the YOLO model's expected input size.
    Args:
        image_path: Path to the input image.
        input_size: Tuple of (width, height) for resizing.
    Returns:
        Preprocessed image and the original image dimensions.
    """
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size  # (width, height)
    resized_image = original_image.resize(input_size, resample=Image.Resampling.BILINEAR)
    input_data = np.expand_dims(np.array(resized_image, dtype=np.float32) / 255.0, axis=0)
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
    heapq.heapify(pq)

    selected_results = []

    while pq:
        _, max_result = heapq.heappop(pq)
        selected_results.append(max_result)

        remaining_results = []
        while pq:
            _, detection = heapq.heappop(pq)
            if box_iou(max_result.rect, detection.rect) < 0.3:# and detection.detection_class == max_result.detection_class:  # IoU threshold
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

    return nms(results)


def get_food(output_data, cls_index):
    # filter out rows with confidence scores lower than the threshold
    food_rects = []
    for row in output_data:
        if row.detection_class in cls_index:
            food_rects.append(row.rect)
    return food_rects 



def postprocess_output(output_data, original_size, threshold=0.3):
    """
    Postprocess the output data to extract bounding boxes, scores, and classes.
    Args:
        output_data: List of output arrays from the TFLite model.
        original_size: Original size of the input image.
        threshold: Confidence threshold to filter predictions.
    Returns:
        List of predictions with (box, score, class_id).
    """
    boxes, scores, classes = output_data
    boxes = boxes[0]  # Assuming the first dimension is batch
    scores = scores[0]
    classes = classes[0]

    predictions = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score > threshold:
            # Rescale the bounding box to the original image size
            y1, x1, y2, x2 = box
            width, height = original_size
            x1, x2 = int(x1 * width), int(x2 * width)
            y1, y2 = int(y1 * height), int(y2 * height)
            predictions.append({"box": (x1, y1, x2, y2), "score": score, "class_id": int(class_id)})
    return predictions


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
    predictions = postprocess(output_data[0], 0.3, 9, input_size)

    # get tray
    crop_bboxes = get_food(predictions, [0, 2, 3, 4, 5] )

    scaleX = 1280 / 224
    scaleY = 720 / 224

    # check if the type of tray_bbox is int
    if len(crop_bboxes) == 0:
        print("no objects found")
        return

    for idx_bbox, each_bbox in enumerate(crop_bboxes):
        left = int(int(each_bbox.left) * scaleX)
        top = int(int(each_bbox.top) * scaleY)
        width = int(int(each_bbox.width()) * scaleX)
        height = int(int(each_bbox.height()) * scaleY)

        # crop the original image using left, top, width, height
        # convert to RGB

        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # convert to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # convert to RGB
        cropped_image = original_image[top:top + height, left:left + width]

        # save cropped image
        if save_intermediate:
            save_cropped_img = Image.fromarray(cropped_image)
            if len(save_root) == 0:
                save_root = os.path.dirname(image_path)
            img_save_path = os.path.join(save_root, os.path.basename(image_path).replace('.jpeg', f'_crop_{idx_bbox}.jpeg'))
            save_cropped_img.save(img_save_path)




if __name__ == "__main__":
    # Paths
    model_path = "/home/kaeun.kim/yolov11/runs/segment-kids-school-maskGT/train15/weights/best_saved_model/best_float16.tflite"  # Replace with your YOLO TFLite model path

    # image paths
    from glob import glob
    webps = glob("/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/train/zero_waste/*.jpeg")
    # webps = ["/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/train/non_zero_waste/busan-gukje-hs_211220_113716_0_L_A_VS-21101803_Trayfile.jpeg"]
    preds = []
    for webp in webps:
        if 'crop' not in webp:
            print(f"==== Processing {webp} ===== ")
            run_full_inference(model_path, webp, save_intermediate=True, save_root='/mnt/nas/data/kaeun/q4/binary_zerowaste/food_binary_original/zero_waste')

    print('')

    # dataset_path = '/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/train/zero_waste'
    # images = glob(dataset_path + '/*.jpeg')


