from pycocotools import coco

coco_info = coco.COCO('/mnt/nas/data/sangmin/testset_trayfile.json')

images = coco_info.loadImgs(coco_info.getImgIds())
kids_imgs = []
kids_anns = []

for image in images:
    file_name = image['file_name']
    image_name = file_name.split('/')[-1].split('.')[0]
    dep = image_name.split('_')[0]
    if dep.endswith('-dc'):

