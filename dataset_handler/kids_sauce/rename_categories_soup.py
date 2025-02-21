import json
data_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-removed-curry.json'

# replace category_id with category_type_id
data_info = json.load(open(data_path, 'r'))
data_info['category_types'].append({'id': 25, 'name': 'soup'})

soup_category_names = [
    '가쓰오장국',
    '갈비탕',
    '갈비탕(*당면)',
    '감자국',
    '감자된장국',
    '감자맑은국',
    '감자양파국',
    '감자양팟국',
    '건새우아욱된장국',
    '곤약어묵국',
    '근대국',
    '근대된장국',
    '김치국',
    '냉이된장국',
    '다시마감자국',
    '다시마무챗국',
    '단호박된장국',
    '달걀국',
    '달걀양팟국',
    '달걀파국',
    '달걀팟국',
    '닭고기대파국',
    '닭곰탕',
    '닭다리백숙',
    '닭안심떡국',
    '동태살맑은국',
    '돼지고기감자탕',
    '돼지고기된장찌개',
    '된장미역국',
    '두부된장국',
    '두부맑은국',
    '두부북어국',
    '두부탕국',
    '들깨감자국',
    '들깨무국',
    '들깨미역국',
    '들깨배추국',
    '들깨시래기국',
    '들깨양배추국',
    '들깨팽이버섯국',
    '떡국',
    '떡만두국',
    '만두국',
    '맑은김치국',
    '맑은느타리버섯국',
    '맑은단호박국',
    '맑은두부국',
    '맑은무국',
    '맑은미역국',
    '맑은순두부국',
    '맑은순두붓국',
    '맑은시금치국',
    '맑은애호박국',
    '맑은유부국',
    '맑은청경채국',
    '맑은콩나물국',
    '맑은콩비지찌개',
    '멸치장국',
    '명란떡국',
    '모듬버섯국',
    '모시조개국',
    '무국',
    '무된장국',
    '무채국',
    '무채된장국',
    '무챗국',
    '물만두국',
    '미소된장국',
    '미소미역된장국',
    '미소장국',
    '미역국',
    '미역된장국',
    '미역미소장국',
    '미역장국',
    '바지락국',
    '바지락미역국',
    '바지락살무국',
    '배추당면국',
    '배추된장국',
    '배추맑은국',
    '버섯국',
    '버섯된장국',
    '버섯맑은국',
    '버섯전골',
    '버섯찌개',
    '봄동된장국',
    '부추달걀국',
    '부추된장국',
    '부추들깨국',
    '부추유부국',
    '북어국',
    '북어미역국',
    '사각어묵국',
    '새우살맑은국',
    '새우살미역국',
    '새우살부춧국',
    '새우살양배추국',
    '새우살탕국',
    '소고기감자양파국',
    '소고기국',
    '소고기당면국',
    '소고기대파국',
    '소고기떡국',
    '소고기무국',
    '소고기무채국',
    '소고기미역국',
    '소고기배춧국',
    '소고기버섯국',
    '소고기양팟국',
    '소고기우거지국',
    '소고기탕국',
    '수제비국',
    '숙주된장국',
    '숙주맑은국',
    '숙주미소된장국',
    '순두부국',
    '순두부된장국',
    '순두부맑은국',
    '순살동태국',
    '순살어묵국',
    '순두부백탕',
    '시금치된장국',
    '시금치맑은국',
    '시금치콩나물된장국',
    '숭늉',
    '시래기국',
    '시래기된장국',
    '시래기맑은국',
    '시레기된장국',
    '실파달걀국',
    '실파된장국',
    '실파우동국물',
    '쑥갓맑은국',
    '아욱국',
    '아욱된장국',
    '안매운김치국',
    '안매운돼지고기김치국',
    '안매운버섯육개장',
    '안매운유부김치국',
    '안매운콩비지찌개',
    '애호박된장국',
    '애호박맑은국',
    '애호박양팟국',
    '애호박채국',
    '양배추된장국',
    '양배추맑은국',
    '양지설렁탕',
    '양파된장국',
    '어묵국',
    '어묵파국',
    '얼갈이된장국',
    '연두부달걀국',
    '열무된장국',
    '열무맑은국',
    '오징어국',
    '오징어무국',
    '오징어양배추국',
    '우거지국',
    '우거지된장국',
    '우거지들깨국',
    '유부된장국',
    '유부맑은국',
    '유부장국',
    '전복미역국',
    '조개살맑은국',
    '조개살미역국',
    '조랭이떡국',
    '짬뽕국',
    '쪽파두붓국',
    '참치미역국',
    '채소당면국',
    '채소된장국',
    '청경채된장국',
    '청경채맑은국',
    '청경채차돌박이된장국',
    '청국장찌개',
    '콩가루배추국',
    '콩나물국',
    '콩나물김치국',
    '콩나물팟국',
    '콩비지국',
    '콩비지찌개',
    '크림스프',
    '팽이버섯된장국',
    '팽이버섯맑은국',
    '팽이버섯장국',
    '팽이장국',
    '해물부춧국',
    '해물순두부찌개',
    '홍합미역국',
    '황태국',
    '황태미역국',
    '황탯국',
]

soup_map_id = {}

for scn in soup_category_names:
    for cat in data_info['categories']:
        if cat['name'] == scn:
            soup_map_id[cat['id']] = scn

for ann in data_info['annotations']:
    if ann['category_id'] in soup_map_id:
        if ann['category_type_id'] != 14: #소스가 아니면
            ann['category_type_id'] = 25

new_anns = []
for ann in data_info['annotations']:
    ann['category_id'] = ann['category_type_id']
    new_anns.append(ann)

data_info['annotations'] = new_anns
data_info['categories'] = data_info['category_types']
del data_info['category_types'] 

# map categories if needed
category_map = {
    9: 6, # mix food -> food
    24: 6, # 흑미밥  -> food
    23: 6, #현미밥 -> food
    22: 6, #찹쌀밥 -> food
    21: 6, # 잔멸치볶음 -> food
    19: 6, # 소고기당면국 -> food
    15: 5,#spoon -> cutlery
    7: 5,#fork -> cutlery
    3: 5, # chopsticks -> cutlery
    18: 6, # 백미밥 -> food
    17: 6, # 닭고기채소볶음밥 -> food
    20: 6, #소고기버섯국 -> food
}


# map the category id using category_map
for ann in data_info['annotations']:
    ann['category_id'] = category_map.get(ann['category_id'], ann['category_id'])

# update categories by removing the ones that are mapped
new_categories = []
for cat in data_info['categories']:
    if cat['id'] in category_map:
        continue
    else:
        new_categories.append(cat)
    
data_info['categories'] = new_categories

# remove categories
remove_category_ids = {13, 2}

# remove images which its annotation has category_id in remove_category_ids
remove_image_ids = set()
for ann in data_info['annotations']:
    if ann['category_id'] in remove_category_ids:
        remove_image_ids.add(ann['image_id'])

new_categories = []
for cat in data_info['categories']:
    if cat['id'] in remove_category_ids:
        continue
    else:
        new_categories.append(cat)
data_info['categories'] = new_categories

data_info['images'] = [img for img in data_info['images'] if img['id'] not in remove_image_ids]
data_info['annotations'] = [ann for ann in data_info['annotations'] if ann['image_id'] not in remove_image_ids]

# update categories index within data_info['categories]
cat_map = {}
for i, cat in enumerate(data_info['categories']):
    cat_map[cat['id']] = i
    cat['id'] = i

# update annotations index within data_info['annotations']
for ann in data_info['annotations']:
    ann['category_id'] = cat_map[ann['category_id']]

with open('/mnt/nas/data/kaeun/2025q1/kids/sauce_food_soup/no-curry-soup-coco.json', 'w+', encoding='utf-8') as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)