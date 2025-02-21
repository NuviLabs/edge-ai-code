from pycocotools import coco
import json

class myCOCO():
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.coco_info = coco.COCO(self.coco_path)

    def get_dates(self):
        images = self.coco_info.dataset['images']
        dates = {}
        for image in images:
            date = image['file_name'].split('/')[-1].split('.')[0].split('_')[1]
            dates[date] = dates.get(date, 0) + 1
        self.dates = dates 

    def get_train_val_test_dates(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1.0")
            
        train_dates = set()
        val_dates = set()
        test_dates = set()
        tot_count = 0
        sum_count = sum(self.dates.values())
        
        for date, count in self.dates.items():
            if count + tot_count > sum_count * (1 - test_ratio):  # Last portion for test
                test_dates.add(date)
            elif count + tot_count > sum_count * train_ratio:  # Middle portion for validation
                val_dates.add(date)
            else:  # First portion for training
                train_dates.add(date)
            tot_count += count
        return train_dates, val_dates, test_dates
    
    def split_train_val_test_date(self, train_dates, val_dates, test_dates):
        images = self.coco_info.dataset['images']
        train_images, val_images, test_images = [], [], []
        train_anns, val_anns, test_anns = [], [], []
        
        for image in images:
            anns = self.coco_info.loadAnns(self.coco_info.getAnnIds(imgIds=image['id']))
            date = image['file_name'].split('/')[-1].split('.')[0].split('_')[1]
            if date in train_dates:
                train_images.append(image)
                train_anns.extend(anns)
            elif date in val_dates:
                val_images.append(image)
                val_anns.extend(anns)
            elif date in test_dates:
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
    coco_path = '/mnt/nas/data/.hubAPI/carleton-university-gt:25.01.16_coco.json'
    mycoco = myCOCO(coco_path)
    mycoco.get_dates()
    train_dates, val_dates, test_dates = mycoco.get_train_val_test_dates()
    train_coco, val_coco, test_coco = mycoco.split_train_val_test_date(
        train_dates=train_dates, 
        val_dates=val_dates,
        test_dates=test_dates
    )
    
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_train.json', train_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_val.json', val_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_test.json', test_coco)