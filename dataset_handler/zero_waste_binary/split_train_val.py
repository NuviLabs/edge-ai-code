import os
from pathlib import Path
import random
from collections import defaultdict

def split_by_department(base_dir='/home/kaeun.kim/kaeun-dev/dataset', val_ratio=0.2):
    # Dictionary to store images by class and department
    class_dept_images = defaultdict(lambda: defaultdict(list))
    
    # Walk through the directory
    for class_name in os.listdir(base_dir):
        class_path = Path(base_dir) / class_name
        if not class_path.is_dir():
            continue
            
        # Get all images and their department info
        for img_path in class_path.glob('**/*.jpeg'):  # adjust extension if needed
            # Extract department name from path
            # Assuming path structure: ./dataset/class_name/dept_name/xxx.jpg
            dept_name = img_path.name.split('/')[-1].split('_')[0]

            class_dept_images[class_name][dept_name].append(str(img_path))

    # Prepare train/val splits
    train_images = defaultdict(list)
    val_images = defaultdict(list)

    # Split by department for each class while considering image counts
    for class_name, dept_dict in class_dept_images.items():
        total_images = sum(len(images) for images in dept_dict.values())
        target_val_images = int(total_images * val_ratio)
        current_val_images = 0
        
        # Sort departments by size for better distribution
        sorted_depts = sorted(dept_dict.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Initialize all departments as training
        for dept, images in sorted_depts:
            train_images[class_name].extend(images)
        
        # Move departments to validation until we reach target ratio
        for dept, images in sorted_depts:
            if current_val_images < target_val_images:
                # Remove from train
                train_images[class_name] = [img for img in train_images[class_name] 
                                          if img not in images]
                # Add to validation
                val_images[class_name].extend(images)
                current_val_images += len(images)
            else:
                break

    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for class_name in class_dept_images.keys():
        print(f"\nClass: {class_name}")
        print(f"Train images: {len(train_images[class_name])}")
        print(f"Val images: {len(val_images[class_name])}")
        
    return train_images, val_images

# Usage example
if __name__ == "__main__":
    train_images, val_images = split_by_department(base_dir='/home/kaeun.kim/kaeun-dev/dataset', val_ratio=0.2)
    # Copy images to train and validation directories
    import shutil
    import os

    # Create train and val directories if they don't exist
    train_dir = '/home/kaeun.kim/kaeun-dev/dataset/train'
    val_dir = '/home/kaeun.kim/kaeun-dev/dataset/dev'
    
    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # Copy train images
    for class_name, images in train_images.items():
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(class_dir, img_name)
            shutil.copy2(img_path, dst_path)
            
    # Copy validation images 
    for class_name, images in val_images.items():
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(class_dir, img_name)
            shutil.copy2(img_path, dst_path)

    
    # Optionally save the splits to files
    # for split_name, split_data in [('train', train_images), ('val', val_images)]:
    #     for class_name, images in split_data.items():
    #         output_file = f'{split_name}_{class_name}.txt'
    #         with open(output_file, 'w') as f:
    #             f.write('\n'.join(images))