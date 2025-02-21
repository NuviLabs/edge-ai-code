from pycocotools import coco
import os
import shutil
from pathlib import Path
import multiprocessing as mp
from itertools import islice

def copy_image_batch(image_batch, output_base_dir, batch_num):
    output_dir = os.path.join(output_base_dir, f'batch_{batch_num}')
    os.makedirs(output_dir, exist_ok=True)
    
    for img in image_batch:
        src_path = os.path.join(img['file_name'])
        dst_path = os.path.join(output_dir, img['file_name'].split('/')[-1])
        shutil.copy2(src_path, dst_path)

def main():
    coco_info = coco.COCO('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/zero_waste_binary/non_zerowaste_school.json')
    
    # Get all images from COCO
    images = list(coco_info.imgs.values())
    
    # Assuming images are in the same directory as the COCO json
    output_base_dir = '/home/kaeun.kim/kaeun-dev/non_zero_waste_binary'  # Change this to your desired output directory
    
    # Split images into batches of 100
    batch_size = len(images)
    image_batches = [list(islice(images, i, i + batch_size)) 
                     for i in range(0, len(images), batch_size)]
    
    # Create process pool and process batches in parallel
    with mp.Pool() as pool:
        pool.starmap(copy_image_batch,
                    [(batch, output_base_dir, i) 
                     for i, batch in enumerate(image_batches)])

if __name__ == '__main__':
    main()

