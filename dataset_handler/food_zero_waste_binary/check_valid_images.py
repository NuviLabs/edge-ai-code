import os
from PIL import Image
from glob import glob
from nuviAPI import s3_api

def get_s3(img_path, buket_name='nuvi-data'):
    filename = os.path.basename(img_path)
    info = filename.split('_')
    dep = info[0]
    date = info[1]
    bld = info[4]
    ba = info[5]
    prefix = f'{dep}/{date}/{bld}/{ba}/{filename}'
    s3_uri = s3_api.make_uri(buket_name, prefix)
    return s3_uri
data_path = '/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original'
imgs = glob(data_path+'/**/*.jpeg', recursive=True)

from_uris = []
to_uris = []
for img in imgs:
    from_uris.append(get_s3(img))
    to_uri = img.split('/')[-1].split('.')[0] + '.png'
    to_uris.append(to_uri)

# remove the images first
for img in to_uris:
    os.remove(img)

s3_api.cp(from_uris, to_uris)
