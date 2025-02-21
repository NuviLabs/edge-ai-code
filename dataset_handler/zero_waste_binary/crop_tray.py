from glob import glob
from pycocotools import coco

images = glob('/home/kaeun.kim/kaeun-dev/non_zero_waste_binary/batch_0/*.jpeg')
coco_data_path = '/mnt/nas/data/.hubAPI/all-tray-new_coco.json'
coco_data = coco.COCO(coco_data_path)

import os
from PIL import Image

# Get all categories and their supercategories
cats = coco_data.loadCats(coco_data.getCatIds())
cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

output_dir = '/home/kaeun.kim/kaeun-dev/non_zero_waste_binary/batch_0_cropped'
os.makedirs(output_dir, exist_ok=True)

for img_path in images:
    # Get image filename
    img_name = os.path.basename(img_path)
    # Check if image has already been cropped
    output_path = os.path.join(output_dir, img_name)
    if os.path.exists(output_path):
        continue

    # Find corresponding image in COCO dataset
    img_info = None
    for img_id in coco_data.getImgIds():
        coco_img = coco_data.loadImgs(img_id)[0]
        if img_name in coco_img['file_name']:
            img_info = coco_img
            break
            
    if img_info is None:
        continue
        
    # Get annotations for this image
    ann_ids = coco_data.getAnnIds(imgIds=img_info['id'])
    anns = coco_data.loadAnns(ann_ids)
    
    # Find tray annotation
    tray_ann = None
    for ann in anns:
        if cat_id_to_supercat[ann['category_id']] == 'tray':
            tray_ann = ann
            break
    
    if tray_ann is None:
        continue
        
    # Get tray bbox coordinates
    x, y, w, h = map(int, tray_ann['bbox'])
    
    # Open and crop image
    img = Image.open(img_path)
    cropped_img = img.crop((x, y, x+w, y+h))
    
    # Save cropped image
    output_path = os.path.join(output_dir, img_name)
    cropped_img.save(output_path)
