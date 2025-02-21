from pycocotools import coco
import json
import random

class myCOCO():
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.coco_info = coco.COCO(self.coco_path)

    def split_images(self, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        total_imgs = self.coco_info.getImgIds()
        random.shuffle(total_imgs)
        train_imgs = total_imgs[:int(len(total_imgs) * train_ratio)]
        val_imgs = total_imgs[int(len(total_imgs) * train_ratio):int(len(total_imgs) * (train_ratio + val_ratio))]
        test_imgs = total_imgs[int(len(total_imgs) * (train_ratio + val_ratio)):]
        print(f"Number of images in train: {len(train_imgs)}")
        print(f"Number of images in validation: {len(val_imgs)}")
        print(f"Number of images in test: {len(test_imgs)}")
        return train_imgs, val_imgs, test_imgs
    
    def get_info_from_imgIds(self, imgIds):
        images = self.coco_info.loadImgs(imgIds)
        anns = self.coco_info.loadAnns(self.coco_info.getAnnIds(imgIds=imgIds))
        return images, anns

    def get_new_coco(self, imgIds):
        images, anns = self.get_info_from_imgIds(imgIds)
        coco_dict = self.coco_info.dataset.copy()
        coco_dict['images'] = images
        coco_dict['annotations'] = anns
        return coco_dict

def save_coco(save_path, coco_dict):
    with open(save_path, 'w+') as f:
        json.dump(coco_dict, f)
    print(f'Saved to {save_path} ...')

        
if __name__ == '__main__':
    coco_path = '/mnt/nas/data/.hubAPI/carleton-university-gt:25.01.16_coco.json'
    mycoco = myCOCO(coco_path)
    train_imgs, val_imgs, test_imgs = mycoco.split_images()
    train_coco, val_coco, test_coco = mycoco.get_new_coco(train_imgs), mycoco.get_new_coco(val_imgs), mycoco.get_new_coco(test_imgs)

    
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_train.json', train_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_val.json', val_coco)
    save_coco('/mnt/nas/data/kaeun/2025q1/carleton/carleton-university-gt:25.01.16_filtered_coco_test.json', test_coco)