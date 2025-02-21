import pandas as pd
from nuviAPI import s3_api
import os

csv_path = '/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/test/QA_image - 시트1.csv'
df = pd.read_csv(csv_path)
partial_df = df[df['덮밥O'] == True]
image_paths = partial_df['s3_uri_after'].tolist()
from_uris = []
to_uris = []
for image_path in image_paths:
    from_uri = image_path.split('https://s3tool.nuvi-lab.com/nuvi-data/')[-1].replace('?file=','/')
    from_uris.append(from_uri)
    to_uri = from_uri.split('/')[-1]
    to_uri = os.path.join('/mnt/nas/data/kaeun/2025q1/kids/testset/images_difficult/', to_uri)
    to_uris.append(to_uri)
csv_path = '/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/test/QA_image - 시트2.csv'
df = pd.read_csv(csv_path)
partial_df = df[df['덮밥O'] == True]
image_paths = partial_df['s3_uri_after'].tolist()
for image_path in image_paths:
    from_uri = 's3://'+image_path.split('https://s3tool.nuvi-lab.com/')[-1].replace('?file=','/')
    from_uris.append(from_uri)
    to_uri = from_uri.split('/')[-1]
    to_uri = os.path.join('/mnt/nas/data/kaeun/2025q1/kids/testset/images_difficult/', to_uri)
    to_uris.append(to_uri)

s3_api.cp(from_uris, to_uris)
print(len(image_paths))