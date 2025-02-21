import os
from glob import glob
import json
from pycocotools import coco

include_imgs_path = '/mnt/nas/data/keunchul/segmentation_yolo_A/images/test'
coco_dataset = '/mnt/nas/data/kaeun/q4/kids/mask:24.09.11_processed_coco_kids.json'
coco_info = coco.COCO(coco_dataset)

# Get list of included image filenames
include_imgs = glob(os.path.join(include_imgs_path, '*.jpeg'))
include_img_names = [os.path.basename(x) for x in include_imgs]

# Initialize new COCO structure
new_coco = {
    'images': [],
    'annotations': [],
    'categories': coco_info.dataset['categories']
}

# Load all images from COCO
all_imgs = coco_info.loadImgs(coco_info.getImgIds())

# Filter images and annotations
for img in all_imgs:
    img_filename = os.path.basename(img['file_name'])
    if img_filename in include_img_names:
        # Add image to new dataset
        new_coco['images'].append(img)
        
        # Get and add corresponding annotations
        ann_ids = coco_info.getAnnIds(imgIds=img['id'])
        anns = coco_info.loadAnns(ann_ids)
        new_coco['annotations'].extend(anns)

# Save new COCO dataset
output_path = '/mnt/nas/data/kaeun/2025q1/kids/mask:24.09.11_processed_coco_kids_test.json'
with open(output_path, 'w+') as f:
    json.dump(new_coco, f)

print(len(include_img_names), len(new_coco['images']))
