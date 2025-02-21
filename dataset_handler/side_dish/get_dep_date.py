# from pycocotools import coco

# coco_path = '/mnt/nas/data/.hubAPI/all-tray-new_coco.json'
# coco_info = coco.COCO(coco_path)

# imgs = coco_info.loadImgs(coco_info.getImgIds())

# for img in imgs:
#     file_name = img['file_name']
#     dep = file_name.split('/')[-1].split('.')[0].split('_')[0]
#     date = file_name.split('/')[-1].split('.')[0].split('_')[1]
#     print('')

from pycocotools import coco
from datetime import datetime

def get_department_dates(coco_path):
    """
    Returns a dictionary where the key is the department and the values are the minimum and maximum dates.
    
    :param coco_path: Path to the COCO JSON file.
    :return: Dictionary with department as the key and (min_date, max_date) as the value.
    """
    # Load COCO annotations
    coco_info = coco.COCO(coco_path)
    imgs = coco_info.loadImgs(coco_info.getImgIds())
    
    # Dictionary to store department and their dates
    department_dates = {}
    
    for img in imgs:
        file_name = img['file_name']
        
        # Extract department and date
        dep = file_name.split('/')[-1].split('.')[0].split('_')[0]
        date_str = file_name.split('/')[-1].split('.')[0].split('_')[1]
        try:
            # Parse date string to a datetime object for comparison
            date_str = '20'+date_str
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            # Skip invalid date formats
            continue
        
        # Update the dictionary with min and max dates
        if dep not in department_dates:
            department_dates[dep] = (date, date)  # Initialize with the first date
        else:
            current_min, current_max = department_dates[dep]
            department_dates[dep] = (min(current_min, date), max(current_max, date))
    
    # Convert datetime objects back to strings for clarity
    final_dict = {}
    for dep in department_dates:
        for k in ['-es', '-ms', '-hs'] and dep not in ['kunkook-university-ms']:
            if k in dep:
                min_date, max_date = department_dates[dep]
                final_dict[dep] = (min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
    
    return final_dict 

final_dict = get_department_dates('/mnt/nas/data/.hubAPI/all-tray-new_coco.json')
import json

with open('final_dict.json', 'w') as f:
    json.dump(final_dict, f)



