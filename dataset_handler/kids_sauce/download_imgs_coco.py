from nuviAPI import s3_api
import json
import os

json_path = '/mnt/nas/data/.hubAPI/kids-mask:25.01.10.json'
data_path = '/mnt/nas/data/.hubAPI'

with open(json_path, 'r') as f:
    data = json.load(f)

from_uris = []
to_uris = []

for img in data['images']:
    png_path = img['file_name'].split('.')[0] + '.png'
    jpeg_path = img['file_name'].split('.')[0] + '.jpeg'
    jpg_path = img['file_name'].split('.')[0] + '.jpg'
    if not os.path.exists(png_path) and not os.path.exists(jpeg_path) and not os.path.exists(jpg_path):
        from_uri = s3_api.make_uri('nuvi-data', img['file_name'].split('.')[0] + '.png')
        from_uris.append(from_uri)
        to_uris.append(os.path.join(data_path, jpeg_path))
# prettyprint
print(f"downloading...{len(from_uris)} images")
s3_api.cp(from_uris, to_uris)