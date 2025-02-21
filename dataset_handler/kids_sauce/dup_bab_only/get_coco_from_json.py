import os
import json
from glob import glob
from tqdm import tqdm

coco_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-only-curry.json'
coco_info = json.load(open(coco_path, 'r'))

def combine_annotations():
    # Directory containing the JSON files
    json_dir = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/empty_dupbab2'
    json_dir_before = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/images/before'
    
    # Get all JSON files in the directory
    json_files = glob(os.path.join(json_dir, '*_annotations.json'))
    json_files_before = glob(os.path.join(json_dir_before, '*_annotations.json'))
    json_files.extend(json_files_before)
    
    if not json_files:
        print("No JSON files found in the directory")
        return
        
    # Initialize the COCO format structure
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': coco_info['categories'],
        'category_types': coco_info['category_types']
    }
    
    # Process each JSON file
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Add image information
        image_info = data['image_info']
        coco_format['images'].append(image_info)
        
        # Add annotations
        coco_format['annotations'].extend(data['annotations'])
            
    # Save the combined COCO format file
    output_file = '/mnt/nas/data/kaeun/2025q1/kids/dupbab/combined_coco_250205_empty_dupbab.json'
    with open(output_file, 'w+') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Combined COCO file created at: {output_file}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Total categories: {len(coco_format['categories'])}")

if __name__ == "__main__":
    combine_annotations()