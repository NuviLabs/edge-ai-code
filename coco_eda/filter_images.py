from pycocotools import coco
import json

class myCOCO():
    def __init__(self, coco_path):
        self.coco_path = coco_path 
        self.coco_info = coco.COCO(self.coco_path)
    
    def save_coco(self, save_path):
        with open(save_path, 'w+') as f:
            json.dump(self.coco_info.dataset, f)
        print(f'Saved to {save_path} ...')

    def remove_images_by_cats(self, rm_cats, cat_key):
        images = self.coco_info.loadImgs(self.coco_info.getImgIds())
        new_images = []
        for image in images:
            skip = False
            anns = self.coco_info.loadAnns(self.coco_info.getAnnIds(image['id']))
            for ann in anns:
                cat_id = ann['category_id']
                cat_info = self.coco_info.loadCats(cat_id)[0]
                if cat_info[cat_key] in rm_cats:
                    skip = True
            if not skip:
                new_images.append(image)
        self.coco_info.dataset['images'] = new_images
    

if __name__ == '__main__':
    coco_path = '/mnt/nas/data/kaeun/q3/all_tray_new_supercategory.json'
    mycoco = myCOCO(coco_path)
    mycoco.remove_images_by_cats(['beverage', 'dessert', 'ignore'], 'supercategory')
    mycoco.save_coco('/mnt/nas/data/kaeun/q3/all_tray_new_supercategory_filtered.json')