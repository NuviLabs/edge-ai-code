from pycocotools import coco
import json
import pandas as pd

class myCOCO():
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.coco_info = coco.COCO(self.coco_path)
    def remove_cats(self, rm_cats, cat_key):
        # cat_key is 'name' or 'supercategory'
        old_cats = self.coco_info.loadCats(self.coco_info.getCatIds())
        new_cats = []
        for old_cat in old_cats:
            if old_cat[cat_key] not in rm_cats:
                new_cats.append(old_cat)
        self.coco_info.dataset['categories'] = new_cats
    def update_cats(self, map_cats, old_key, new_key):
        # old_key, new_key are 'name' or 'supercategory'
        old_cats = self.coco_info.loadCats(self.coco_info.getCatIds())
        new_cats = []
        for old_cat in old_cats:
            new_cat = old_cat.copy()
            new_cat[new_key] = map_cats[old_cat[old_key]]
            new_cats.append(new_cat)
        self.coco_info.dataset['categories'] = new_cats

    def group_cats(self, map_cats, cat_key):
        # cat_key is 'name' or 'supercategory'
        old_cats = self.coco_info.loadCats(self.coco_info.getCatIds())
        new_cats = []
        for old_cat in old_cats:
            new_cat = old_cat.copy()
            for key, val in map_cats.items():
                if old_cat[cat_key] in val:
                    new_cat[cat_key] = key
            new_cats.append(new_cat)
        self.coco_info.dataset['categories'] = new_cats

    def save_coco(self, save_path):
        with open(save_path, 'w+') as f:
            json.dump(self.coco_info.dataset, f)
        print(f'Saved to {save_path} ...')

def get_map_cats(path_to_csv):
    df = pd.read_csv(path_to_csv)
    # iterate though each row
    map_cats = {}
    for index, row in df.iterrows():
        map_cats[row['name']] = row['new_name']
    return map_cats


if __name__ == '__main__':
    coco_path = '/mnt/nas/data/.hubAPI/ediya:25.01.02_coco.json'
    mycoco = myCOCO(coco_path)
    # map_cats = get_map_cats('/mnt/nas/data/sangmin/map_cats.csv')
    group_cats = {
        'food': {'food', 'fruit', 'kimchi', 'rice', 'rice_etc', 'soup'},
        'tray': {'tray'},
        'cutlery': {'cutlery'},
        'hand': {'hand'},
        'bone': {'bone'},
        'peel': {'peel'},
        'package': {'package'},
        'sauce': {'sauce'},
        'tray': {'tray'},
    }
    mycoco.group_cats(group_cats, 'supercategory')
    output_path = '/mnt/nas/data/kaeun/q3/all_tray_new_supercategory_filtered_merged.json'
    mycoco.save_coco(output_path)