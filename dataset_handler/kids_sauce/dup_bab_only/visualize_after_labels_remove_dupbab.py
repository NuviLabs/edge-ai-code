from glob import glob
import os
import json
import cv2
import numpy as np

json_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-only-curry.json'
json_info = json.load(open(json_path, 'r'))
data_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/empty_dupbab'
images = glob(data_path+ '/*.jpeg')


remove_category = [
    '새우카레라이스', '깐풍돼지고기덮밥', '돼지고기카레라이스', '일본식닭고기덮밥', '쇠불고기덮밥', 
    '돈부리덮밥', '안매운마파두부덮밥', '닭살짜장밥', '소고기배추덮밥', '짜장밥', 
    '바몬드카레라이스소스', '채소짜장밥', '카레', '감자채카레볶음', '굴소스소고기덮밥', 
    '소고기카레라이스', '돼지고기깻잎덮밥', '카레라이스', '새우살짜장밥', '짜장소스', 
    '닭살카레라이스', '마파두부덮밥', '짜장덮밥', '카레소스', '소고기카레소스', 
    '연근카레라이스', '브로콜리카레라이스소스', '돼지고기카레', '김치볶음덮밥', '짜장', 
    '닭바베큐덮밥'
]

# remove_category_type_ids = {17, 18, 19, 20, 21, 22, 23}
remove_category_type_ids = {}



# 1) Find category IDs to remove
remove_category_ids = set()
for cat in json_info['categories']:
    if cat['name'] in remove_category:
        remove_category_ids.add(cat['id'])

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
            if ann['category_id'] in remove_category_ids:
                continue
            anns.append(ann)
    return anns, img_dict

# Create output directory for annotated images and json files
output_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/empty_dupbab2'
os.makedirs(output_path, exist_ok=True)

for image in images:
    image_path = os.path.join('/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after', os.path.basename(image))
    img = cv2.imread(image_path)
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
