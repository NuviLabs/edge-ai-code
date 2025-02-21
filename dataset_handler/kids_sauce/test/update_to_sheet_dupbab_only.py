
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
from collections import defaultdict
import os
from run_tflite import run_entire_pipeline

scope = ["https://spreadsheets.google.com/feeds"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("/home/kaeun.kim/kaeun-dev/nuvilab/qa/kids_detector/nuvilab-dab12266f615.json", scope)
client = gspread.authorize(credentials)
sheet_service = googleapiclient.discovery.build('sheets', 'v4', credentials=credentials)
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ/edit?gid=0#gid=0")
spreadsheet_id = "1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ"
worksheet = spreadsheet.get_worksheet(0)  # Replace with the desired worksheet index
webps = glob("/mnt/nas/data/kaeun/q4/kids/detector_qa/**/*.png", recursive=True)
matched_pairs = defaultdict(dict)
sheet_name = "덮밥only"

def get_all_values(spreadsheet_id, sheet_name):
    result = sheet_service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
    return result.get('values', [])

df = get_all_values(spreadsheet_id, sheet_name)
model_path = "/home/kaeun.kim/ultralytics/runs/segment/train/weights/best_saved_model/best_float16.tflite"
from tqdm import tqdm
for row_idx, row in enumerate(tqdm(df)):
    after_name = row[3]
    after_url = after_name.replace('/s3://nuvi-data', '')
    if not after_name.startswith("http"):
        continue
    after_name = after_name.replace('?file=', '/').split('/')[-1]
    dep = after_name.split('/')[-1].split('_')[0]
    date = after_name.split('/')[-1].split('_')[1]
    after_path = f'/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab_test_images/{after_name}'
    if not os.path.exists(after_path):
        print(f"after_path not found: {after_path}")
        continue
    after_result = run_entire_pipeline(model_path, after_path, num_cls=12, food_index=4, tray_index=10)

    before_name = row[2]
    before_url = before_name.replace('/s3://nuvi-data', '')
    if not before_name.startswith("http"):
        continue
    before_name = before_name.replace('?file=', '/').split('/')[-1]
    dep = before_name.split('/')[-1].split('_')[0]
    date = before_name.split('/')[-1].split('_')[1]
    before_path = f'/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab_test_images/{before_name}'
    if not os.path.exists(before_path):
        print(f"before_path not found: {before_path}")
        continue
    before_result = run_entire_pipeline(model_path, before_path, num_cls=12, food_index=4, tray_index=10)

    if before_result == 0:
        intake = 0
    else:
        intake = (1 - (after_result/before_result))*100
    # cell2 = worksheet.cell(row_idx+1, 3)
    # cell2.value = before_url 
    # cell3 = worksheet.cell(row_idx+1, 4)
    # cell3.value = after_url 
    cell7 = worksheet.cell(row_idx+1, 8)
    cell7.value = intake 


    # worksheet.update_cells([cell2, cell3, cell7], value_input_option="USER_ENTERED")
    worksheet.update_cells([cell7], value_input_option="USER_ENTERED")
    time.sleep(8)
