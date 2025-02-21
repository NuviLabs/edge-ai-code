import json
import os
from pycocotools import mask as mask_utils
import numpy as np
from glob import glob

data_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-250205_empty_dupbab.json'
data_info = json.load(open(data_path, 'r'))

def get_box_area(bbox):
    return bbox[2] * bbox[3]

def boxes_overlap(bbox1, bbox2):
    x1_1, y1_1 = bbox1[0], bbox1[1]
    x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x1_2, y1_2 = bbox2[0], bbox2[1]
    x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    return not (x1_1 > x2_2 or x1_2 > x2_1 or y1_1 > y2_2 or y1_2 > y2_1)

def merge_rle_masks(rle1, rle2, img_height, img_width):
    # Decode RLE to binary masks
    mask1 = mask_utils.decode(rle1)
    mask2 = mask_utils.decode(rle2)
    
    # Merge masks (logical OR operation)
    merged_mask = np.logical_or(mask1, mask2)
    
    # Encode back to RLE
    merged_rle = mask_utils.encode(np.asfortranarray(merged_mask))
    return {
        'size': merged_rle['size'],
        'counts': merged_rle['counts'].decode('utf-8')  # Convert bytes to string
    }
# filter out annotations which has area less than 2000
new_anns = []
for ann in data_info['annotations']:
    if ann['area'] < 2000:
        continue
    else:
        new_anns.append(ann)

# filter out annotations which the segmentation pixel area is does not take up more than 50% of the bounding box area
new_anns = []
for ann in data_info['annotations']:
    bounding_box_area = ann['bbox'][2] * ann['bbox'][3]
    if ann['area'] / bounding_box_area < 0.3:
        continue
    else:
        new_anns.append(ann)

new_anns2 = []
# print progress
from tqdm import tqdm

skip_imgs = glob('/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after_annotated_250124/*.jpeg')
skip_imgs = set([os.path.basename(img).split('.')[0] for img in skip_imgs])
for img_idx, img in enumerate(tqdm(data_info['images'])):
    img_id = img['id']
    anns = [ann for ann in new_anns if ann['image_id'] == img_id]
    if '_B_' in img['file_name']:
        new_anns2.extend(anns)
        continue
    if img['file_name'].split('/')[-1].split('.')[0] in skip_imgs:
        new_anns2.extend(anns)
        continue


    class_4_anns = [ann for ann in anns if ann['category_id'] == 4]  # food
    class_9_anns = [ann for ann in anns if ann['category_id'] == 9]  # sauce
    
    merged_anns = []
    processed_ann_ids = set()
    
    for ann4 in class_4_anns:
        for ann9 in class_9_anns:
            if boxes_overlap(ann4['bbox'], ann9['bbox']):
                area4 = get_box_area(ann4['bbox'])
                area9 = get_box_area(ann9['bbox'])
                
                # If sauce area is larger and food area is significantly smaller
                # (e.g., food area is less than 30% of sauce area)
                if area9 > area4 and (area4 / area9) < 0.3:
                    # Merge masks
                    merged_rle = merge_rle_masks(
                        ann4['rle'], 
                        ann9['rle'],
                        img['height'],
                        img['width']
                    )
                    
                    # Create new annotation with merged mask
                    merged_ann = ann9.copy()  # Use sauce annotation as base
                    merged_ann['rle'] = merged_rle
                    merged_ann['area'] = float(mask_utils.area(merged_rle))
                    
                    merged_anns.append(merged_ann)
                    processed_ann_ids.add(ann4['id'])
                    processed_ann_ids.add(ann9['id'])
                elif area9 > area4 and (area4 / area9) > 0.9:
                    merged_anns.append(ann4)
                    processed_ann_ids.add(ann4['id'])
                    processed_ann_ids.add(ann9['id'])
                else:
                    # Keep original annotations if they don't meet merge criteria
                    if ann4['id'] not in processed_ann_ids:
                        merged_anns.append(ann4)
                    if ann9['id'] not in processed_ann_ids:
                        merged_anns.append(ann9)
    
    # Add any remaining annotations that weren't involved in overlaps
    for ann in anns:
        if ann['id'] not in processed_ann_ids:
            merged_anns.append(ann)
    
    new_anns2.extend(merged_anns)


data_info['annotations'] = new_anns2
print(len(new_anns2))

with open('/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered-250205-empty-dupbab.json', 'w+', encoding='utf-8') as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)