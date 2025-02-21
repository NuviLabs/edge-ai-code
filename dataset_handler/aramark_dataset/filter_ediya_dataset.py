import json
from pycocotools import coco
json_info = json.load(open('/mnt/nas/data/.hubAPI/ediya:25.01.06_coco.json'))
coco_info = coco.COCO('/mnt/nas/data/.hubAPI/ediya:25.01.06_coco.json')
# coco_info = coco.COCO('/mnt/nas/data/.hubAPI/coco_2q-mask-labeling.json')

dates = [
'231220',
'231219',
'231215',
'231214',
'231213',
'231212',
'231211',
'231205',
'231207',
'231206',
'231208',
'231204',
'231031',
'231030',
'231027',
'231026',
'231025',
'231024',
'231023',
'231020',
'231103',
'231101',
'231102',
'231229',
'231228',
'231227',
'231222',
'231221',
'231117',
'231116',
'231115',
'231114',
'231110',
'231113',
'231109',
'231106',
'231107',
'231108',
]
new_anns = []
new_cats_id = []
new_imgs = []

old_imgs = coco_info.loadImgs(coco_info.getImgIds())
for img in old_imgs:
    imgdate = img['file_name'].split('/')[-1].split('.')[0].split('_')[1]
    if imgdate in dates:
        anns = coco_info.loadAnns(coco_info.getAnnIds(imgIds=img['id']))
        curr_new_cats_id = [ann['category_type_id'] for ann in anns]
        if 15 in curr_new_cats_id:
            continue
        new_cats_id += [ann['category_type_id'] for ann in anns]
        new_imgs.append(img)
        # swap category id with category_type_id
        for ann in anns:
            ann['category_id'] = ann['category_type_id']
            new_anns.append(ann)
    
new_cats_id = set(new_cats_id)
new_cats = [cat for cat in json_info['category_types'] if cat['id'] in new_cats_id ]

# save new coco
coco_info.dataset['images'] = new_imgs
coco_info.dataset['annotations'] = new_anns
coco_info.dataset['categories'] = new_cats
with open('/mnt/nas/data/kaeun/2025q1/aramark/ediya:25.01.06_filtered_coco.json', 'w') as f:
    json.dump(coco_info.dataset, f)
print()
