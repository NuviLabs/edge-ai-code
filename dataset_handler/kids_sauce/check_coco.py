import json
path = '/mnt/nas/data/kaeun/2025q1/kids/sauce_food_soup/no-curry-soup-train.json'

data_info = json.load(open(path, 'r'))

existing_cats = set()
for ann in data_info['annotations']:
    existing_cats.add(ann['category_id'])
for cat in existing_cats:
    print(data_info['categories'][cat]['name'])