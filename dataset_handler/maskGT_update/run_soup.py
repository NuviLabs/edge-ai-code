from pycocotools import coco
from ultralytics import YOLO
import numpy as np
import cv2
import os
import json

coco_info = coco.COCO('/mnt/nas/data/kaeun/q4/maskGT/soup_mixfood_maskGT/updated_coco_v2.json')
model = YOLO('/home/kaeun.kim/kaeun-dev/soup_model.pt')

imgs = coco_info.getImgIds()

coco_info.dataset['categories'].append({'id': 9, 'name': 'soup', 'supercategory': None})

def calculate_iou(box1, box2):
    # Convert COCO format [x,y,w,h] to [x1,y1,x2,y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

new_anns = []

for img_id in imgs:
    # Load image info and annotations
    img_info = coco_info.loadImgs(img_id)[0]
    img_path = os.path.join('/mnt/nas/data/.hubAPI', img_info['file_name'])  # Adjust path if needed
    img = cv2.imread(img_path)
    
    # Get YOLO predictions
    results = model(img)
    yolo_boxes = results[0].boxes
    
    # Get existing annotations for this image
    ann_ids = coco_info.getAnnIds(imgIds=img_id)
    anns = coco_info.loadAnns(ann_ids)
    
    # Compare each annotation with YOLO predictions
    for ann in anns:
        coco_box = ann['bbox']  # [x, y, w, h]
        
        for yolo_box in yolo_boxes:
            # Convert YOLO box to COCO format
            x1, y1, x2, y2 = yolo_box.xyxy[0].cpu().numpy()
            yolo_box_coco = [x1, y1, x2-x1, y2-y1]
            
            iou = calculate_iou(coco_box, yolo_box_coco)
            
            if iou > 0.9:
                # Update category to 'soup'
                # You'll need to specify the category ID for 'soup'
                ann['category_id'] = 9  # Replace with actual category ID
    new_anns.extend(anns)

# # Save updated annotations
coco_info.dataset['annotations'] = new_anns

with open('/mnt/nas/data/kaeun/q4/maskGT/soup_mixfood_maskGT/soup_mixfood_maskGT.json', 'w+') as f:
    json.dump(coco_info.dataset, f)


