import json
import os

def merge_coco_files(file1_path, file2_path, output_path):
    # Load both JSON files
    with open(file1_path, 'r') as f1:
        data1 = json.load(f1)
    with open(file2_path, 'r') as f2:
        data2 = json.load(f2)

    # Initialize merged data structure
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': [],
        'category_types': []
    }

    # compare if any of the image ids in data1 is in data2
    for img in data1['images']:
        if img['id'] in [img2['id'] for img2 in data2['images']]:
            print(f"Image ID {img['id']} is in both datasets")

    # compare if any of the annotation ids in data1 is in data2
    for ann in data1['annotations']:
        if ann['id'] in [ann2['id'] for ann2 in data2['annotations']]:
            print(f"Annotation ID {ann['id']} is in both datasets")

    # Merge images from first file (no changes needed)
    merged_data['images'].extend(data1['images'])
    merged_data['images'].extend(data2['images'])
    merged_data['annotations'].extend(data1['annotations'])
    merged_data['annotations'].extend(data2['annotations'])

    # Merge categories from first file
    merged_data['categories'].extend(data1['categories'])

    # Merge category types from first file
    merged_data['category_types'].extend(data1['category_types'])

    # Save merged data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"Merged dataset statistics:")
    print(f"Number of images: {len(merged_data['images'])}")
    print(f"Number of annotations: {len(merged_data['annotations'])}")
    print(f"Number of categories: {len(merged_data['categories'])}")
    print(f"Number of category types: {len(merged_data['category_types'])}")

if __name__ == "__main__":
    # Example usage
    file1_path = "/mnt/nas/data/kaeun/2025q1/kids/dupbab/combined_coco_250205_empty_dupbab.json"
    file2_path = "/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-removed-curry.json"
    output_path = "/mnt/nas/data/growth_zero/kids_sauce_food/merged_coco_raw_250205_empty_dupbab.json"
    
    merge_coco_files(file1_path, file2_path, output_path)