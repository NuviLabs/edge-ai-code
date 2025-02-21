path1 = '/home/kaeun.kim/kaeun-dev/zero_waste_binary/batch_0'
path2 = '/home/kaeun.kim/kaeun-dev/zero_waste_binary2/batch_0'

import os
import shutil

# Get list of existing files in path1
existing_files = set(os.listdir(path1))

# Move files from path2 to path1, skipping duplicates
for filename in os.listdir(path2):
    if filename not in existing_files:
        src = os.path.join(path2, filename)
        dst = os.path.join(path1, filename)
        shutil.move(src, dst)
