import json
data_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-removed-curry.json'

# replace category_id with category_type_id
data_info = json.load(open(data_path, 'r'))
new_anns = []
for ann in data_info['annotations']:
    ann['category_id'] = ann['category_type_id']
    new_anns.append(ann)

data_info['annotations'] = new_anns
data_info['categories'] = data_info['category_types']
del data_info['category_types'] 

# map categories if needed
category_map = {
    9: 6, # mix food -> food
    24: 6, # 흑미밥  -> food
    23: 6, #현미밥 -> food
    22: 6, #찹쌀밥 -> food
    21: 6, # 잔멸치볶음 -> food
    19: 6, # 소고기당면국 -> food
    15: 5,#spoon -> cutlery
    7: 5,#fork -> cutlery
    3: 5, # chopsticks -> cutlery
    18: 6, # 백미밥 -> food
    17: 6, # 닭고기채소볶음밥 -> food
    20: 6, #소고기버섯국 -> food
}

# map the category id using category_map
for ann in data_info['annotations']:
    ann['category_id'] = category_map.get(ann['category_id'], ann['category_id'])

# update categories by removing the ones that are mapped
new_categories = []
for cat in data_info['categories']:
    if cat['id'] in category_map:
        continue
    else:
        new_categories.append(cat)
    
data_info['categories'] = new_categories

# remove categories
remove_category_ids = {13, 2, 14}

# remove images which its annotation has category_id in remove_category_ids
remove_image_ids = set()
for ann in data_info['annotations']:
    if ann['category_id'] in remove_category_ids:
        remove_image_ids.add(ann['image_id'])

new_categories = []
for cat in data_info['categories']:
    if cat['id'] in remove_category_ids:
        continue
    else:
        new_categories.append(cat)
data_info['categories'] = new_categories

data_info['images'] = [img for img in data_info['images'] if img['id'] not in remove_image_ids]
data_info['annotations'] = [ann for ann in data_info['annotations'] if ann['image_id'] not in remove_image_ids]

# update categories index within data_info['categories]
cat_map = {}
for i, cat in enumerate(data_info['categories']):
    cat_map[cat['id']] = i
    cat['id'] = i

# update annotations index within data_info['annotations']
for ann in data_info['annotations']:
    ann['category_id'] = cat_map[ann['category_id']]

with open('/mnt/nas/data/kaeun/2025q1/kids/no-sauce/no-curry-coco.json', 'w+', encoding='utf-8') as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)