import json
import os
import shutil
from tqdm import tqdm
json_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-only-curry.json'
with open(json_path, 'r') as f:
    data = json.load(f)

before_images = []
after_images = []
for img in data['images']:
    if '_B_' in img['file_name']:
        before_images.append(img['file_name'])
    else:
        after_images.append(img['file_name'])

from_path = '/mnt/nas/data/.hubAPI/'
to_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/before'

for img in tqdm(before_images):
    img_from_path = os.path.join(from_path, img)
    img_to_path = os.path.join(to_path, img.split('/')[-1])
    os.makedirs(os.path.dirname(img_to_path), exist_ok=True)
    shutil.copy(img_from_path, img_to_path)

to_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after'
for img in tqdm(after_images):
    img_from_path = os.path.join(from_path, img)
    img_to_path = os.path.join(to_path, img.split('/')[-1])
    os.makedirs(os.path.dirname(img_to_path), exist_ok=True)
    shutil.copy(img_from_path, img_to_path)

print(before_images)
print(after_images)