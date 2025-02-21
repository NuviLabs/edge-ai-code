from glob import glob
from ultralytics import YOLO 
import json

output_dir = '/home/kaeun.kim/kaeun-dev/android_output/manual_crop'
# images = glob('/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/**/*.jpeg')
images = ['/home/kaeun.kim/kaeun-dev/android_output/zero_0.00_1.00_0.7194822_1732783435968.webp']

import os
from PIL import Image

# Get all categories and their supercategories

model = YOLO('/home/kaeun.kim/yolov11/runs/segment-kids-school-maskGT/train15/weights/best.pt')
for image in images:
    filename = os.path.basename(image)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if already processed
    # if os.path.exists(output_path):
    #     continue
        
    # Run inference
    result = model(image)[0]
    
    boxes = result.boxes
    tray_idx = -1
    tray_conf = -1

    if len(boxes) > 0:
        clses = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        for ic, (confidence, label) in enumerate(zip(confs, clses)):
            if label == 6.0:  # Adjust class ID as per your YOLO training
                if confidence > tray_conf:
                    tray_conf = confidence
                    tray_idx = ic
        

        # Get the tray bounding box
        x1, y1, x2, y2 = boxes.xyxy[tray_idx].cpu().numpy()
    
        # Open and crop image
        img = Image.open(image)
        cropped_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # Save cropped image
        cropped_img.save(output_path)
        print(f'Saved cropped image: {filename}')

# run inference using that image and save the annotations of yolo
