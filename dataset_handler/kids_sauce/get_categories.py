import json
coco_path = '/mnt/nas/data/.hubAPI/kids-mask:25.01.10.json'

with open(coco_path, 'r') as f:
    data = json.load(f)

# save categories into csv
with open('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/categories.csv', 'w+') as f:
    for category in data['categories']:
        f.write(f"{category['id']},{category['name']}\n")