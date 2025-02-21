from pycocotools import coco
import pandas as pd
import json
import random
from collections import defaultdict

random.seed(1997)

coco_data_path = '/mnt/nas/data/.hubAPI/all-tray-new_coco.json'
coco_data = coco.COCO(coco_data_path)
new_images = []
new_categories = []
new_anns = []

# only has one tray
# tray is a certain size
# no __ignore__ supercategory
# no food supercategory
# can have etc supercategory
# Get all categories and their supercategories
cats = coco_data.loadCats(coco_data.getCatIds())
cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

# Dictionary to store images by department
dept_images = {}
dept_anns = {}
dept_cats = {}

filtered_images_kids = {}
filtered_images_school = {}


for img_id in coco_data.getImgIds():
    img = coco_data.loadImgs(img_id)[0]
    ann_ids = coco_data.getAnnIds(imgIds=img_id)
    anns = coco_data.loadAnns(ann_ids)
    
    # Check if any annotation has __ignore__ supercategory
    has_ignore = any(cat_id_to_supercat[ann['category_id']] == '__ignore__' for ann in anns)
    if has_ignore:
        continue
        
    # Check if image has at least one tray annotation
    num_tray = sum([1 for ann in anns if cat_id_to_supercat[ann['category_id']] == 'tray'])
    if not num_tray == 1:
        continue
    # Find the tray annotation
    tray_ann = next(ann for ann in anns if cat_id_to_supercat[ann['category_id']] == 'tray')
    tray_bbox = tray_ann['bbox']  # [x, y, width, height]
    
    # Check if any food annotation is within the tray bbox
    num_food = 0 
    for ann in anns:
        if cat_id_to_supercat[ann['category_id']] == 'food':
            food_bbox = ann['bbox']
            
            # Check if food bbox is within tray bbox
            if (food_bbox[0] >= tray_bbox[0] and 
                food_bbox[1] >= tray_bbox[1] and
                food_bbox[0] + food_bbox[2] <= tray_bbox[0] + tray_bbox[2] and 
                food_bbox[1] + food_bbox[3] <= tray_bbox[1] + tray_bbox[3]):
                num_food +=1 

    if '-dc' in img['file_name']:
        filtered_images_kids.setdefault(num_food, []).append(img_id)
    elif any( suffix in img['file_name'] for suffix in ('-es', '-ms', '-hs')):
        filtered_images_school.setdefault(num_food, []).append(img_id)

# Save filtered images to JSON files
import json
import os

# Create output directory if it doesn't exist
os.makedirs('filtered_images', exist_ok=True)

# Save kids filtered images
with open('./filtered_images_kids.json', 'w') as f:
    json.dump(filtered_images_kids, f)

# Save school filtered images  
with open('./filtered_images_school.json', 'w') as f:
    json.dump(filtered_images_school, f)
