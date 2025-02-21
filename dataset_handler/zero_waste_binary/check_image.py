from pycocotools import coco

coco_info = coco.COCO('/mnt/nas/data/.hubAPI/all-tray-new_coco.json')

import cv2
import numpy as np
import os

def visualize_category_samples(category_name, output_dir='sample_images', num_samples=10):
    """
    Visualizes and saves sample images containing instances of a specified category
    
    Args:
        category_name (str): Name of the category to visualize
        output_dir (str): Directory to save the output images
        num_samples (int): Number of sample images to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get category ID from name
    cat_ids = coco_info.getCatIds(catNms=[category_name])
    if not cat_ids:
        print(f"Category '{category_name}' not found")
        return
        
    # Get images containing this category
    img_ids = coco_info.getImgIds(catIds=cat_ids)
    if not img_ids:
        print(f"No images found containing category '{category_name}'")
        return
        
    # Limit to num_samples
    img_ids = img_ids[:num_samples]
    
    for i, img_id in enumerate(img_ids):
        # Load image info and annotations
        img_info = coco_info.loadImgs(img_id)[0]
        ann_ids = coco_info.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco_info.loadAnns(ann_ids)
        
        # Read image
        img = cv2.imread(img_info['file_name'])
        if img is None:
            print(f"Could not read image: {img_info['file_name']}")
            continue
            
        # Draw bounding boxes
        for ann in anns:
            bbox = ann['bbox']  # [x,y,width,height]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(img, category_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save image
        output_path = os.path.join(output_dir, f'{category_name}_{i+1}.jpg')
        cv2.imwrite(output_path, img)
        print(f'Saved {output_path}')

visualize_category_samples('nuvi_cfg', './')