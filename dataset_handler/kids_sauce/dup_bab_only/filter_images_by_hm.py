import pandas as pd 
import json
from tqdm import tqdm
from glob import glob
import os

data_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered-250205-empty-dupbab.json'
csv_path = '/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/image_hm.csv'

skip_images_path = "/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/after_annotated_250124"
skip_images = set([file.split('/')[-1].split('.')[0] for file in glob(os.path.join(skip_images_path, "*.jpeg"))])

df = pd.read_csv(csv_path)
# select only images that are not in skip_images
df = df[~df['file_name'].str.split('/').str[-1].str.split('.').str[0].isin(skip_images)]
file_names = [file.split('/')[-1].split('.')[0] for file in df['file_name'].tolist()]
image_ids = set(df['image_id'].tolist())

with open(data_path, 'r') as f:
    data = json.load(f)

print(len(data['images']))
data['images'] = [img for img in tqdm(data['images']) if img['id'] not in image_ids]
print(len(data['images']))

with open('/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered-250205-empty-dupbab-hm.json', 'w+') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
