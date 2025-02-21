import pandas as pd 
import json
from tqdm import tqdm

data_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered.json'
csv_path = '/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/image_hm.csv'

df = pd.read_csv(csv_path)
image_ids = set(df['image_id'].tolist())

with open(data_path, 'r') as f:
    data = json.load(f)

print(len(data['images']))
data['images'] = [img for img in tqdm(data['images']) if img['id'] not in image_ids]
print(len(data['images']))

with open('/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered-hm.json', 'w+') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
