import json

json_info = json.load(open('/mnt/nas/data/.hubAPI/kids-mask:25.02.19_coco.json'))
for cat in json_info['categories']:
    print(cat['name'])
