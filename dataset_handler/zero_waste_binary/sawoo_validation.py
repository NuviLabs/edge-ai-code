import pandas as pd
from nuviAPI import s3_api
import os

info = pd.read_csv('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/zero_waste_binary/on_device_ai_based_score_202411261656.csv')

info['food_pixel_ratio'] = info['food_pixel'] / info['tray_area']
filtered_rows = info[(info['food_pixel_ratio'] >= 0.15) & (info['food_pixel_ratio'] < 0.3)]
user_ids = filtered_rows['user_id'].tolist()

prefix='sawoo-es/241126/L/A/'

imgs = s3_api.list_objects('nuvi-data', prefixes=prefix, endswith='png')
from_uris = []
to_uris = []
for img in imgs:
    for user_id in user_ids:
        if '_' + str(user_id) + '_' in img:
            from_uri = s3_api.make_uri('nuvi-data', img)
            from_uris.append(from_uri)
            to_uri = os.path.join('/mnt/nas/data/kaeun/q4/binary_zerowaste/binary_sawoo_1126', from_uri.split('/')[-1])
            to_uris.append(to_uri)

s3_api.cp(from_uris=from_uris, to_uris=to_uris)

print('')