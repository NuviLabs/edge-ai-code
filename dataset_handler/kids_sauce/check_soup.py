import json
data_path = '/mnt/nas/data/.hubAPI/kids-mask:25.01.10.json'

# replace category_id with category_type_id
data_info = json.load(open(data_path, 'r'))
soup_category_names = (
    # '고구마순살닭볶음탕',
    '닭곰탕',
    '닭다리백숙',
    # '닭볶음탕',
    # '돈사태찜',
    # '돼지갈비찜',
    # '돼지고기갈비찜',
    # '돼지고기감자찜',
    '돼지고기감자탕',
    # '로제찜닭',
    '맑은콩비지찌개',
    '버섯전골',
    '버섯찌개',
    # '뿌리채소돼지갈비찜',
    '순두부백탕',
    '숭늉',
    '실파우동국물',
    # '안매운닭볶음탕',
    # '안매운돼지고기김치찜',
    '안매운버섯육개장',
    '안매운콩비지찌개',
    '양지설렁탕',
    '청국장찌개',
    # '콩나물돼지고기찜',
    '콩비지찌개',
    '크림스프',
    '해물순두부찌개',
)

for cat in data_info['categories']:
    if cat['name'] in soup_category_names:
        for img in data_info['images']:
            exists = False
            anns = [ann for ann in data_info['annotations'] if ann['image_id'] == img['id']]
            for ann in anns:
                if data_info['categories'][ann['category_id']]['name'] == cat['name']:
                    exists = True
                    break
            if exists == True:
                print(f"{cat['name']} exists in {img['file_name']}")
                break