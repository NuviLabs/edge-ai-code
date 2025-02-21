
from pycocotools import coco
import pandas as pd
import json
import random
from collections import defaultdict
import os

random.seed(1998)

coco_data_path = '/mnt/nas/data/.hubAPI/all-tray-new_coco.json'
coco_data = coco.COCO(coco_data_path)

def sample_equally(data, keys, num_samples):
    # Group images by department and date
    dept_date_groups = defaultdict(list)
    total_imgs = []
    for key in keys:
        if str(key) in data:
            total_imgs.extend(data[str(key)])
    
    for img in total_imgs:
        img_name = coco_data.loadImgs(img)[0]['file_name']
        info = img_name.split('/.hubAPI/')[-1]
        dep = info.split('/')[0]
        date = info.split('/')[1]
        key = (dep, date)  # Group by department and month
        dept_date_groups[key].append(img)
    
    # Calculate samples per group
    num_groups = len(dept_date_groups)
    samples_per_group = num_samples // num_groups
    remaining = num_samples % num_groups
    
    selected_images = []
    
    # Sample from each group
    for group_images in dept_date_groups.values():
        num_to_select = min(
            samples_per_group + (1 if remaining > 0 else 0),
            len(group_images)
        )
        
        if num_to_select > 0:
            selected = random.sample(group_images, num_to_select)
            selected_images.extend(selected)
            
        remaining -= 1
    
    # If we still need more images, sample from remaining pool
    if len(selected_images) < num_samples:
        remaining_pool = [img for img in total_imgs if img not in selected_images]
        if remaining_pool:
            additional_needed = num_samples - len(selected_images)
            additional = random.sample(remaining_pool, min(additional_needed, len(remaining_pool)))
            selected_images.extend(additional)
    
    return selected_images

def create_coco(img_ids, coco_data, save_dir):
    imgs = coco_data.loadImgs(img_ids)
    anns = coco_data.loadAnns(coco_data.getAnnIds(img_ids))
    coco_data.dataset['images'] = imgs
    coco_data.dataset['annotations'] = anns
    # Save COCO dataset to JSON file
    with open(save_dir, 'w+') as f:
        json.dump(coco_data.dataset, f)



# zero_waste = sample_equally(json.load(open('/home/kaeun.kim/kaeun-dev/filtered_images_school.json')), [0], 500)
# create_coco(zero_waste, coco_data, './zerowaste_school_2.json')

non_zero_waste = sample_equally(json.load(open('/home/kaeun.kim/kaeun-dev/filtered_images_school.json')), list(range(1, 31)), 1500)
create_coco(non_zero_waste, coco_data, './non_zerowaste_school.json')