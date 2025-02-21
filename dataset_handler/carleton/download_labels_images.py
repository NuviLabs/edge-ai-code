from nuviAPI import s3_api
import os

dep = 'carleton-university'
dates = ['241111/L', '241116/L', '241118/L']
from_uris = []
to_uris = []

save_path = '/mnt/nas/data/kaeun/q4/carleton-university/'
for date in dates:
    date_files = s3_api.list_objects(bucket_name='nuvi-data', prefixes=f'{dep}/{date}')
    for each_file in date_files:
        if '.png' in each_file:
            from_uri = s3_api.make_uri('nuvi-data', each_file)
            from_uris.append(from_uri)
            save_path_this = os.path.join(save_path, 'images')
            filename = os.path.basename(each_file)
            to_uris.append(os.path.join(save_path_this, filename))
        elif 'Trayfile.json' in each_file:
            from_uri = s3_api.make_uri('nuvi-data', each_file)
            from_uris.append(from_uri)
            save_path_this = os.path.join(save_path, 'annotations')
            filename = os.path.basename(each_file)
            to_uris.append(os.path.join(save_path_this, filename))

s3_api.cp(from_uris, to_uris)