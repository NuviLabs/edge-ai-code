import json
from collections import defaultdict

data_path = '/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_train.json'
data_info = json.load(open(data_path, 'r'))

# for each category type, I want to count the number of images that have that category type
category_type_count = defaultdict(set)
for ann in data_info['annotations']:
    category_type_count[ann['category_type_id']].add(ann['image_id'])

for k, v in category_type_count.items():
    print(f"{data_info['category_types'][k]['name']} ({data_info['category_types'][k]['id']}): {len(v)}")