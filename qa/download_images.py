from nuviAPI import s3_api
import random
import os
import numpy as np

random.seed(1997)

bucket_name = 'nuvi-data'
departments = ['gukbangdaehakyo-dc']
# departments += ['toreatorea-dc', 'moarae-dc', 'joongang-dc', 'dotori-sopoong-nexonbyeol-dc']

# dates = ['241226', '241224', '241223', '241220', '241219']
# dates = ['241213', '241214', '241215', '241216']
dates = ['241017', '241018']
imgs_count = 0
 
img_save_path = '/mnt/nas/data/kaeun/q4/kids/detector_qa/odd_tray'

paths = []
weights = []

for department in departments:
    dep_path = os.path.join(img_save_path, department)
    if not os.path.exists(dep_path):
        os.makedirs(dep_path)

    for date in dates:
        date_path = os.path.join(dep_path, date)
        if not os.path.exists(date_path):
            os.makedirs(date_path)
        prefix = f'{department}/{date}/L/A/'
        imgs = s3_api.list_objects(bucket_name, prefix, endswith='.png')
        imgs_count += len(imgs)
        for img in imgs:
            paths.append({
                'department': department,
                'date': date,
                'nas_path': os.path.join(date_path, img.split('/')[-1]),
                's3_uri': s3_api.make_uri(bucket_name, img)
            })
            weights.append(len(imgs))

# Normalize weights
weights = np.array(weights)
weights = weights / weights.sum()

# Perform weighted random sampling
sampled_paths = random.choices(paths, weights=weights, k=len(paths))

# print the number of images for each department and date
for department in departments:
    for date in dates:
        count = len([path for path in sampled_paths if path['department'] == department and path['date'] == date])
        print(f'{department}/{date}: {count}')

# Now `sampled_paths` contains the sampled images considering the number of images for each department and date

from_uris = []
to_uris = []

for path in sampled_paths:
    from_uris.append(path['s3_uri'])
    to_uris.append(path['nas_path'])

s3_api.cp(from_uris, to_uris)
print(f'{imgs_count} images are downloaded to {img_save_path}')

