from nuviAPI import s3_api

import pymysql
import pandas as pd
import os


DB_HOST = "db-nuvilab.coaxvb1s16kq.ap-northeast-2.rds.amazonaws.com"
DB_USER = "ai-server"
DB_PASSWORD = 'zE-{RbWCqjN)g7@7'
DB_DB = 'kids'


conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_DB, charset='utf8')

department_id = 487
department_name = "wiryeforest-es"
zero_waste_score = 1
date = "2024-12-18"
query = f"select user_id, DATE(created_at), score from kids.edge_ai_intake_score eais;"
cursor = conn.cursor()
cursor.execute(query)
result = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

cursor.close()

df = pd.DataFrame(result, columns=columns)

user_ids = df.user_id.tolist()

date_string = date.replace('-', '')[2:]
imgs = s3_api.list_objects('nuvi-data', f'{department_name}/{date_string}/L/A', endswith='png')

# create csv of user_id and image path and tantan_score
for user_id in user_ids:
    for img in imgs:
        if f'_{user_id}_' in img:
            from_uri = s3_api.make_uri('nuvi-data', img)

            break


df['food_pixel/tray_area'] = df.food_pixel/df.tray_area
if zero_waste_score == 1:
    df2 = df[df['food_pixel/tray_area'] < 0.02]
elif zero_waste_score == 0.75:
    df2 = df[(df['food_pixel/tray_area'] < 0.15) & (df['food_pixel/tray_area'] >= 0.02)]
elif zero_waste_score == 0.5:
    df2 = df[(df['food_pixel/tray_area'] < 0.3) & (df['food_pixel/tray_area'] >= 0.15)]
elif zero_waste_score == 0.25:
    df2 = df[df['food_pixel/tray_area'] >= 0.3]

seg_user_id = set(df2.user_id.tolist())

save_root = "/mnt/nas/data/kaeun/q4/binary_zerowaste/qa/wiryeforest-es"
from_uris = []
to_uris = []
zero_judgement_final_save_path = os.path.join(save_root, 'zero_judgement')
segmentation_final_save_path = os.path.join(save_root, 'segmentation')
if not os.path.exists(zero_judgement_final_save_path):
    os.makedirs(zero_judgement_final_save_path)
if not os.path.exists(segmentation_final_save_path):
    os.makedirs(segmentation_final_save_path)

for user_id in user_ids:
    if user_id in seg_user_id:
        final_save_path = segmentation_final_save_path
    else:
        final_save_path = segmentation_final_save_path
    image_save_path = os.path.join(final_save_path, str(zero_waste_score))
    for img in imgs:
        if f'_{user_id}_' in img:
            from_uri = s3_api.make_uri('nuvi-data', img)
            from_uris.append(from_uri)
            to_uri = os.path.join(image_save_path, img.split('/')[-1])
            to_uris.append(to_uri)
            break


s3_api.cp(from_uris, to_uris)
print(f"Saved to {final_save_path} ... {len(from_uris)} images saved")
