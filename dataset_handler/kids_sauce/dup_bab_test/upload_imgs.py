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
from glob import glob


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

def get_pairs(img_uris):
    pairs = {}
    for img_uri in img_uris:
        img_path = img_uri.split('/')[-1]
        dep = img_path.split('_')[0]
        date = img_path.split('_')[1]
        user_id = img_path.split('_')[3]
        ba = img_path.split('_')[5]
        key = f'{dep}/{date}/{user_id}'
        if key in pairs:  
            pairs[key][ba] = img_uri
        else:
            pairs[key] = {ba: img_uri}
    return pairs

test_imgs = glob('/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab_test_images/*.png')
test_pairs = get_pairs(test_imgs)

for idx, (key, v) in enumerate(test_pairs.items()):
    if len(v) < 2:
        continue
    else:
        after_name = v['A']
        before_name = v['B']
        img_path = after_name.split('/')[-1]
        dep = img_path.split('_')[0]
        date = img_path.split('_')[1]
        bld = img_path.split('_')[4]
        s3_key_before = s3_api.make_uri('nuvi-data', f'{dep}/{date}/{bld}/B/{before_name.split("/")[-1]}')
        s3_key_after = s3_api.make_uri('nuvi-data', f'{dep}/{date}/{bld}/A/{after_name.split("/")[-1]}')
        uuid_before = str(uuid.uuid4())
        cropped_image_key_before = f"kids_qa3/{uuid_before}.png"
        uuid_after = str(uuid.uuid4())
        cropped_image_key_after = f"kids_qa3/{uuid_after}.png"
        cropped_image_url_cloutfront_before = f"https://d3e1nmmw7sjsqu.cloudfront.net/{cropped_image_key_before}"  # Replace with your image URL
        cropped_image_url_cloutfront_after = f"https://d3e1nmmw7sjsqu.cloudfront.net/{cropped_image_key_after}"  # Replace with your image URL
        image_value_before = f'=image("{cropped_image_url_cloutfront_before}")'
        image_value_after = f'=image("{cropped_image_url_cloutfront_after}")'
        s3tool_url_before = f"https://s3tool.nuvi-lab.com/nuvi-data/{s3_key_before.split('nuvi-data/')[-1]}".replace(f'B/{dep}', f'B?file={dep}')
        s3tool_url_after = f"https://s3tool.nuvi-lab.com/nuvi-data/{s3_key_after.split('nuvi-data/')[-1]}".replace(f'A/{dep}', f'A?file={dep}')
        cropped_image_url_before = f'https://s3tool.nuvi-lab.com/nuvi-depth/public/{cropped_image_key_before}'.replace(f'kids_qa3/', f'kids_qa3?file=')
        cropped_image_url_after = f'https://s3tool.nuvi-lab.com/nuvi-depth/public/{cropped_image_key_after}'.replace(f'kids_qa3/', f'kids_qa3?file=')
        to_s3_key_before = s3_api.make_uri('nuvi-depth', f"public/{cropped_image_key_before}")
        to_s3_key_after = s3_api.make_uri('nuvi-depth', f"public/{cropped_image_key_after}")
        s3_api.cp([s3_key_before, s3_key_after],[to_s3_key_before, to_s3_key_after])


        cell1 = worksheet.cell(idx+2, 1)  
        cell1.value = image_value_before  
        cell2 = worksheet.cell(idx+2, 2)  
        cell2.value = image_value_after  
        cell3 = worksheet.cell(idx+2, 3)
        cell3.value = s3tool_url_before
        cell4 = worksheet.cell(idx+2, 4)
        cell4.value = s3tool_url_after
        worksheet.update_cells([cell1, cell2, cell3, cell4 ], value_input_option="USER_ENTERED")
        time.sleep(8)