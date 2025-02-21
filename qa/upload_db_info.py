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
spreadsheet_id = "1WKAA84Q-hxApxSbftt7RtoinMC3r6NZ12avZ0gpybOQ"
worksheet = spreadsheet.get_worksheet(1)  # Replace with the desired worksheet index

sheet_name = "시트2"

from nuviAPI import s3_api

import pymysql
import pandas as pd
import os


DB_HOST = "db-nuvilab.coaxvb1s16kq.ap-northeast-2.rds.amazonaws.com"
DB_USER = "ai-server"
DB_PASSWORD = 'zE-{RbWCqjN)g7@7'
DB_DB = 'kids'


conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_DB, charset='utf8')


def get_all_values(spreadsheet_id, sheet_name):
    result = sheet_service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
    return result.get('values', [])

df = get_all_values(spreadsheet_id, sheet_name)
for row_idx, row in enumerate(df):
    if row_idx <= 330:
        continue
    after_name = row[3]
    if not after_name.startswith("http"):
        continue
    date = after_name.split('/')[-1].split('_')[1]
    # change to 'YYMMDD' to 'YYYY-MM-DD'
    date = f"20{date[:2]}-{date[2:4]}-{date[4:]}"
    user_id = after_name.split('/')[-1].split('_')[3]

    query = f"select score from kids.edge_ai_intake_score eais where user_id={int(user_id)} and date(created_at)='{date}';"
    # query = f"select user_id, DATE(created_at), score from kids.edge_ai_intake_score eais limit 10;"
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    cursor.close()

    df = pd.DataFrame(result, columns=columns)
    old_score = float(df['score'][0])
    print(f"old_score: {old_score}")

    cell13 = worksheet.cell(row_idx+1, 13)
    cell13.value = old_score 


    worksheet.update_cells([cell13], value_input_option="USER_ENTERED")
    time.sleep(8)