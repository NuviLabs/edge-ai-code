from pycocotools import coco
import json

class myCOCO():
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.coco_info = coco.COCO(self.coco_path)

    def get_deps(self):
        images = self.coco_info.dataset['images']
        deps = {}
        for image in images:
            dep = image['file_name'].split('/')[-1].split('.')[0].split('_')[0]
            deps[dep] = deps.get(dep, 0) + 1
        self.deps = deps
    
    def get_train_val_test_deps(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1.0")
            
        train_deps = set()
        val_deps = set()
        test_deps = set()
        tot_count = 0
        sum_count = sum(self.deps.values())
        
        for dep, count in self.deps.items():
            if count + tot_count > sum_count * (1 - test_ratio):  # Last portion for test
                test_deps.add(dep)
            elif count + tot_count > sum_count * train_ratio:  # Middle portion for validation
                val_deps.add(dep)
            else:  # First portion for training
                train_deps.add(dep)
            tot_count += count
        return train_deps, val_deps, test_deps
    
    def split_train_val_test_dep(self, train_deps, val_deps, test_deps):
        images = self.coco_info.dataset['images']
        train_images, val_images, test_images = [], [], []
        train_anns, val_anns, test_anns = [], [], []
        
        for image in images:
            anns = self.coco_info.loadAnns(self.coco_info.getAnnIds(imgIds=image['id']))
            dep = image['file_name'].split('/')[-1].split('.')[0].split('_')[0]
            if dep in train_deps:
                train_images.append(image)
                train_anns.extend(anns)
            elif dep in val_deps:
                val_images.append(image)
                val_anns.extend(anns)
            elif dep in test_deps:
                test_images.append(image)
                test_anns.extend(anns)
        
        print(f"Number of images in train: {len(train_images)}")
        print(f"Number of images in validation: {len(val_images)}")
        print(f"Number of images in test: {len(test_images)}")
        
        train_coco = self.coco_info.dataset.copy()
        train_coco['images'] = train_images
        train_coco['annotations'] = train_anns
        
        val_coco = self.coco_info.dataset.copy()
        val_coco['images'] = val_images
        val_coco['annotations'] = val_anns
        
        test_coco = self.coco_info.dataset.copy()
        test_coco['images'] = test_images
        test_coco['annotations'] = test_anns
        
        return train_coco, val_coco, test_coco

def save_coco(save_path, coco_dict):
    with open(save_path, 'w+') as f:
        json.dump(coco_dict, f)
    print(f'Saved to {save_path} ...')

        
if __name__ == '__main__':
    coco_path = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/merged-coco-soup-filtered-250205-empty-dupbab-hm.json'
    mycoco = myCOCO(coco_path)
    mycoco.get_deps()
    train_deps, val_deps, test_deps = mycoco.get_train_val_test_deps()
    train_coco, val_coco, test_coco = mycoco.split_train_val_test_dep(
        train_deps=train_deps, 
        val_deps=val_deps,
        test_deps=test_deps
    )
    
    save_coco('/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab-soup-hm-coco-train-250205-empty-dupbab.json', train_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab-soup-hm-coco-val-250205-empty-dupbab.json', val_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/kids/dupbab/dupbab-soup-hm-coco-test-250205-empty-dupbab.json', test_coco)