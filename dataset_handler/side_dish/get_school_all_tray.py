from pycocotools import coco

coco_info = coco.COCO('/mnt/nas/data/.hubAPI/all-tray-new_coco.json')
imgs = coco_info.loadImgs(coco_info.getImgIds())
final_imgs = []

for img in imgs:
    dep = img['file_name'].split('/')[-1].split('.')[0].split('_')[0]
    for k in ['-es', '-ms', '-hs']:
        if dep not in ['kunkook-university-ms'] and k in dep:
            final_imgs.append(img['file_name'])

import csv

with open('final_imgs.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name"])
    for img in final_imgs:
        writer.writerow([img])

print()