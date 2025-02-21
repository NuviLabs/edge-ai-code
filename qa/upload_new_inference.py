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
import os


scope = ["https://spreadsheets.google.com/feeds"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("/home/kaeun.kim/kaeun-dev/nuvilab/qa/kids_detector/nuvilab-dab12266f615.json", scope)
client = gspread.authorize(credentials)
sheet_service = googleapiclient.discovery.build('sheets', 'v4', credentials=credentials)
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ/edit?gid=0#gid=0")
spreadsheet_id = "1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ"
worksheet = spreadsheet.get_worksheet(1)  # Replace with the desired worksheet index

sheet_name = "QA"


def get_all_values(spreadsheet_id, sheet_name):
    result = sheet_service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
    return result.get('values', [])

df = get_all_values(spreadsheet_id, sheet_name)
model_path = "/home/kaeun.kim/yolov11/runs/segment/train4/weights/best_saved_model/best_float16.tflite"  # Replace with your YOLO TFLite model path
for row_idx, row in enumerate(df):
    # if row_idx <= 205:
    #     continue
    after_name = row[3]
    before_name = row[2]
    if not after_name.startswith("http"):
        continue

    # after_path = os.path.join('/mnt/nas/data/kaeun/q4/kids/detector_qa/', after_name.split('nuvi-data')[-1].replace('?file=', '/'))
    dep = after_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1].split('_')[0]
    img_date = after_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1].split('_')[1]
    after_path = os.path.join('/mnt/nas/data/kaeun/q4/kids/detector_qa/', dep, img_date, after_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1])
    if not os.path.exists(after_path):
        after_path = f"/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab_test_images/{after_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1]}"
    before_path = os.path.join('/mnt/nas/data/kaeun/q4/kids/detector_qa/', dep, img_date, before_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1])
    if not os.path.exists(before_path):
        before_path = f"/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab_test_images/{before_name.split('nuvi-data')[-1].replace('?file=', '/').split('/')[-1]}"
    pred, zw, label_exists, tray_area_after, food_area_after = run_entire_pipeline_after(model_path, after_path)
    if pred == -1:
        continue
    food_tray_ratio_after = food_area_after / tray_area_after
    pred, zw, label_exists, tray_area_before, food_area_before = run_entire_pipeline_before(model_path, before_path)
    food_tray_ratio_before = food_area_before / tray_area_before
    # before_path = os.path.join('/mnt/nas/data/kaeun/q4/kids/detector_qa/', before_name.split('nuvi-data')[-1].replace('?file=', '/'))
    # print(after_path, before_path)
    # change to 'YYMMDD' to 'YYYY-MM-DD'


    if food_tray_ratio_before == 0:
        intake = 0
    else:
        intake = (1 - (food_tray_ratio_after/food_tray_ratio_before))*100

    cell15 = worksheet.cell(row_idx+1, 16)
    cell15.value = intake
    worksheet.update_cells([cell15], value_input_option="USER_ENTERED")

    # cell15 = worksheet.cell(row_idx+1, 15)
    # cell15.value = food_tray_ratio_before 
    # cell16 = worksheet.cell(row_idx+1, 16)
    # cell16.value = food_tray_ratio_after 
    # worksheet.update_cells([cell15, cell16], value_input_option="USER_ENTERED")



    time.sleep(8)