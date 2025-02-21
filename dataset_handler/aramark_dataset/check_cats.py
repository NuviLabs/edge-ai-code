from pycocotools import coco
coco_info = coco.COCO('/mnt/nas/data/kaeun/2025q1/aramark/ediya:25.01.06_filtered_coco.json')
cats = coco_info.loadCats(coco_info.getCatIds())
for cat in cats:
    print(cat)
