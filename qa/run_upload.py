from final_zero_waste import run_entire_pipeline_before, run_entire_pipeline_after
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from nuviAPI.s3 import s3_api
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from skimage import draw
import uuid
from glob import glob
import googleapiclient.discovery


scope = ["https://spreadsheets.google.com/feeds"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("/home/kaeun.kim/kaeun-dev/nuvilab/qa/kids_detector/nuvilab-dab12266f615.json", scope)
client = gspread.authorize(credentials)
sheet_service = googleapiclient.discovery.build('sheets', 'v4', credentials=credentials)
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ/edit?gid=0#gid=0")
worksheet = spreadsheet.get_worksheet(1)  # Replace with the desired worksheet index
from collections import defaultdict
webps = glob("/mnt/nas/data/kaeun/q4/kids/detector_qa/**/*.png", recursive=True)
matched_pairs = defaultdict(dict)
for webp in webps:
    if 'odd_tray' in webp:
        continue
    dep = webp.split('/')[-1].split('_')[0]
    date = webp.split('/')[-1].split('_')[1]
    user_id = webp.split('/')[-1].split('_')[3]
    if '_A_' in webp:
        matched_pairs[f'{dep}/{date}/{user_id}']['A'] = webp
    elif '_B_' in webp:
        matched_pairs[f'{dep}/{date}/{user_id}']['B'] = webp

num_imgs = 0
model_path = "/home/kaeun.kim/yolov11/runs/segment/train4/weights/best_saved_model/best_float16.tflite"  # Replace with your YOLO TFLite model path
for idx, (k, v) in enumerate(matched_pairs.items()):
    if idx < 90:
        continue
    if len(v) < 2:
        continue
    else:
        num_imgs += 1
        after_name = v['A']
        before_name = v['B']
        img_path = after_name.split('/')[-1]
        dep = img_path.split('_')[0]
        date = img_path.split('_')[1]
        bld = img_path.split('_')[4]
        s3_key_before = s3_api.make_uri('nuvi-data', f'{dep}/{date}/{bld}/B/{before_name.split("/")[-1]}')
        s3_key_after = s3_api.make_uri('nuvi-data', f'{dep}/{date}/{bld}/A/{after_name.split("/")[-1]}')
        pred, zw, label_exists, tray_area_after, food_area_after = run_entire_pipeline_after(model_path, after_name)
        pred, zw, label_exists, tray_area_before, food_area_before = run_entire_pipeline_before(model_path, before_name)
        if pred == -1:
            print("no tray: ", webp)
            continue
        uuid_before = str(uuid.uuid4())
        cropped_image_key_before = f"kids_qa2/{uuid_before}.png"
        uuid_after = str(uuid.uuid4())
        cropped_image_key_after = f"kids_qa2/{uuid_after}.png"
        cropped_image_url_cloutfront_before = f"https://d3e1nmmw7sjsqu.cloudfront.net/{cropped_image_key_before}"  # Replace with your image URL
        cropped_image_url_cloutfront_after = f"https://d3e1nmmw7sjsqu.cloudfront.net/{cropped_image_key_after}"  # Replace with your image URL
        image_value_before = f'=image("{cropped_image_url_cloutfront_before}")'
        image_value_after = f'=image("{cropped_image_url_cloutfront_after}")'
        s3tool_url_before = f"https://s3tool.nuvi-lab.com/nuvi-data/{s3_key_before.split('nuvi-data/')[-1]}".replace(f'B/{dep}', f'B?file={dep}')
        s3tool_url_after = f"https://s3tool.nuvi-lab.com/nuvi-data/{s3_key_after.split('nuvi-data/')[-1]}".replace(f'A/{dep}', f'A?file={dep}')
        cropped_image_url_before = f'https://s3tool.nuvi-lab.com/nuvi-depth/public/{cropped_image_key_before}'.replace(f'kids_qa2/', f'kids_qa2?file=')
        cropped_image_url_after = f'https://s3tool.nuvi-lab.com/nuvi-depth/public/{cropped_image_key_after}'.replace(f'kids_qa2/', f'kids_qa2?file=')
        to_s3_key_before = s3_api.make_uri('nuvi-depth', f"public/{cropped_image_key_before}")
        to_s3_key_after = s3_api.make_uri('nuvi-depth', f"public/{cropped_image_key_after}")
        pred, zw, label_exists, tray_area_after, food_area_after = run_entire_pipeline_after(model_path, after_name)
        if pred == -1:
            continue
        food_tray_ratio_after = food_area_after / tray_area_after
        pred, zw, label_exists, tray_area_before, food_area_before = run_entire_pipeline_before(model_path, before_name)
        food_tray_ratio_before = food_area_before / tray_area_before
        if pred == -1:
            continue
        s3_api.cp([s3_key_before, s3_key_after],[to_s3_key_before, to_s3_key_after])


        cell1 = worksheet.cell(num_imgs+1, 1)  
        cell1.value = image_value_before  
        cell2 = worksheet.cell(num_imgs+1, 2)  
        cell2.value = image_value_after  
        cell3 = worksheet.cell(num_imgs+1, 3)
        cell3.value = s3tool_url_before
        cell4 = worksheet.cell(num_imgs+1, 4)
        cell4.value = s3tool_url_after 
        cell5 = worksheet.cell(num_imgs+1, 5)
        cell5.value = food_tray_ratio_before
        cell6 = worksheet.cell(num_imgs+1, 6)
        cell6.value = food_tray_ratio_after 
        cell7 = worksheet.cell(num_imgs+1, 7)
        cell7.value = uuid_before
        cell8 = worksheet.cell(num_imgs+1, 8)
        cell8.value = uuid_after 
        cell9 = worksheet.cell(num_imgs+1, 9)
        cell9.value = before_name
        cell10 = worksheet.cell(num_imgs+1, 10)
        cell10.value = after_name 


    worksheet.update_cells([cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10], value_input_option="USER_ENTERED")
    time.sleep(10)

   