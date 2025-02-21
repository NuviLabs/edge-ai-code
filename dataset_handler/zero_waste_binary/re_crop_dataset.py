import os
from glob import glob
import shutil

from_dataset_path = '/home/kaeun.kim/kaeun-dev/dataset'

mode = ['train', 'val']
cls = ['non_zero_waste', 'zero_waste']

output_dir = '/mnt/nas/data/kaeun/q4/binary_zerowaste/dataset_original'

images = glob(from_dataset_path + '/train/**/*.jpeg') + glob(from_dataset_path+'/val/**/*.jpeg')

for image in images:
    if 'train' in image or 'val' in image:
        png_name = image.split('/')[-1]
        info = png_name.split('_')
        dep = info[0]
        date = info[1]
        bld = info[4]
        ba = info[5]
        output_img_path = f'/mnt/nas/data/.hubAPI/{dep}/{date}/{bld}/{ba}/{png_name}'

        # get cls
        img_cls = image.split('/')[-2]
        #get mode
        img_mode = image.split('/')[-3]

        save_img_path = os.path.join(output_dir, img_mode, img_cls, png_name)
        if not os.path.exists(os.path.join(output_dir, img_mode)):
            os.mkdir(os.path.join(output_dir, img_mode))
            if not os.path.exists(os.path.join(output_dir, img_mode, img_cls)):
                os.mkdir(os.path.join(output_dir, img_mode, img_cls))

        shutil.copyfile(output_img_path, save_img_path)

