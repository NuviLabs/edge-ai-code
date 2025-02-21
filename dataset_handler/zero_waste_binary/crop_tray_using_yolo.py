from glob import glob
from ultralytics import YOLO 
import json
import os

output_dir = '/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_cropped'
# images = glob('/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original/**/*.jpeg', recursive=True)
images = ['/home/kaeun.kim/kaeun-dev/android_output/zero_images/zero_0.25_0.25_2.6354688E-11_1732683689479.webp']

import os
from PIL import Image

# Get all categories and their supercategories

model = YOLO('/home/kaeun.kim/yolov11/runs/segment-kids-school-maskGT/train15/weights/best.pt')
for image in images:
    filename = os.path.basename(image)
    image_mode = image.split('/')[-3]
    image_cls = image.split('/')[-2]
    output_path = os.path.join(output_dir, image_mode, image_cls, filename)
    if not os.path.exists(os.path.join(output_dir, image_mode, image_cls)):
        os.makedirs(os.path.join(output_dir, image_mode, image_cls), exist_ok=True)
    

    # Skip if already processed
    if os.path.exists(output_path):
        continue
        
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
