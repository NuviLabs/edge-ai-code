from glob import glob
import os
import json
import cv2
import numpy as np

json_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-only-curry.json'
json_info = json.load(open(json_path, 'r'))
data_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after_original'
images = glob(data_path+ '/*.jpeg')

def get_annotation(image_name):
    anns = []
    img_id = None
    img_dict = None
    for img in json_info['images']:
        if image_name in img['file_name']:
            img_id = img['id']
            img_dict = img
            break
    for ann in json_info['annotations']:
        if ann['image_id'] == img_id:
            anns.append(ann)
    return anns, img_dict

# Create output directory for annotated images and json files
output_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after_original'
os.makedirs(output_path, exist_ok=True)

for image in images:
    img = cv2.imread(image)
    annotation, img_dict = get_annotation(image.split('/')[-1])
    
    # Save annotation and img_dict to JSON file
    image_basename = os.path.splitext(os.path.basename(image))[0]
    json_data = {
        'image_info': img_dict,
        'annotations': annotation,
        'modified': 0,
    }
    json_output_file = os.path.join(output_path, f'{image_basename}_annotations.json')
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved JSON annotations: {json_output_file}")
    
    # Draw each annotation on the image
    for ann in annotation:
        # Get bbox coordinates
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add category label and annotation ID
        category_id_type = ann['category_type_id']
        annotation_id = ann['id']
        # Find category name from categories
        # Place text at lower left corner of bbox with both category name and annotation ID
        label_text = f"{category_id_type} (id:{annotation_id})"
        cv2.putText(img, label_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the annotated image
    output_file = os.path.join(output_path, os.path.basename(image))
    cv2.imwrite(output_file, img)
    print(f"Saved annotated image: {output_file}")
