from pycocotools import coco
import pandas as pd
import json

coco_data_path = '/mnt/nas/data/.hubAPI/all-tray-new_coco.json'
coco_data = coco.COCO(coco_data_path)

soups = pd.read_csv('/home/kaeun.kim/kaeun-dev/all_soup.csv')
#convert single column to list
soups = set(soups['0'].tolist())

# iterate through each images and their according annotations
new_anns = []
new_imgs = []
for img_id in coco_data.getImgIds():
    img = coco_data.loadImgs(img_id)[0]
    ann_ids = coco_data.getAnnIds(imgIds=img_id)
    anns = coco_data.loadAnns(ann_ids)
    image_new_anns = []
    for ann in anns:
        # replace the label with the new label
        # get category name
        cat_id = ann['category_id']
        cat_name = coco_data.loadCats(cat_id)[0]['name']
        if cat_name in soups:
            ann['category_id'] = 0
            image_new_anns.append(ann)
        else:
            continue

    if len(image_new_anns) > 0:
        print(f'Image {img_id} has {len(image_new_anns)} soup annotations')
        new_anns.extend(image_new_anns)
        new_imgs.append(img)

new_cats = [{'super_category': 'soup', 'id': 0, 'name': 'soup'}]

# save the new annotations
coco_data.dataset['images'] = new_imgs
coco_data.dataset['categories'] = new_cats
coco_data.dataset['annotations'] = new_anns

# save new coco file
with open('/mnt/nas/data/kaeun/q4/soup_all_tray.json', 'w+') as f:
    json.dump(coco_data.dataset, f)
