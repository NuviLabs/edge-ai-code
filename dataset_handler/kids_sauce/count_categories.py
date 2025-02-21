import json
from collections import defaultdict

data_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-removed-curry.json'
data_info = json.load(open(data_path, 'r'))

# for each category type, I want to count the number of images that have that category type
category_type_count = defaultdict(set)
for ann in data_info['annotations']:
    category_type_count[ann['category_type_id']].add(ann['image_id'])

for k, v in category_type_count.items():
    print(f"{data_info['category_types'][k]['name']} ({data_info['category_types'][k]['id']}): {len(v)}")