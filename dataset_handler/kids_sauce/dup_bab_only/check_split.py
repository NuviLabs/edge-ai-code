from glob import glob
import os
import json

image_paths = "/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after_annotated_250124"
image_names = set([os.path.basename(path).split('.')[0] for path in glob(os.path.join(image_paths, "*.jpeg"))])
train_path = "/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab-soup-hm-coco-test.json"
train_data = json.load(open(train_path, 'r'))

num_imgs = 0
for image in train_data['images']:
    if image['file_name'].split('/')[-1].split('.')[0] in image_names:
        num_imgs += 1
print(f"num_imgs: {num_imgs}")

