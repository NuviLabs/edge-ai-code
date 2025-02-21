import json

json_path = '/mnt/nas/data/.hubAPI/kids-mask:25.01.10.json'
with open(json_path, 'r') as f:
    data = json.load(f)

full_img_ids = set([i['id'] for i in data['images']])

remove_category = [
    '새우카레라이스', '깐풍돼지고기덮밥', '돼지고기카레라이스', '일본식닭고기덮밥', '쇠불고기덮밥', 
    '돈부리덮밥', '안매운마파두부덮밥', '닭살짜장밥', '소고기배추덮밥', '짜장밥', 
    '바몬드카레라이스소스', '채소짜장밥', '카레', '감자채카레볶음', '굴소스소고기덮밥', 
    '소고기카레라이스', '돼지고기깻잎덮밥', '카레라이스', '새우살짜장밥', '짜장소스', 
    '닭살카레라이스', '마파두부덮밥', '짜장덮밥', '카레소스', '소고기카레소스', 
    '연근카레라이스', '브로콜리카레라이스소스', '돼지고기카레', '김치볶음덮밥', '짜장', 
    '닭바베큐덮밥'
]

# remove_category_type_ids = {17, 18, 19, 20, 21, 22, 23}
remove_category_type_ids = {}



# 1) Find category IDs to remove
remove_category_ids = set()
for cat in data['categories']:
    if cat['name'] in remove_category:
        remove_category_ids.add(cat['id'])

# 2) Find images containing these categories
removed_images = set()
completely_remove_images = set()

for anno in data['annotations']:
    if anno['category_type_id'] in remove_category_type_ids:
        completely_remove_images.add(anno['image_id'])
    if anno['category_id'] in remove_category_ids:
        removed_images.add(anno['image_id'])

other_images = full_img_ids - removed_images - completely_remove_images

# 3) Create two separate datasets
removed_data = {'images': [], 'annotations': [], 'categories': data['categories'], 'category_types': data['category_types']}
kept_data = {'images': [], 'annotations': [], 'categories': [], 'category_types': data['category_types']}

# Split images
for img in data['images']:
    if img['id'] in removed_images:
        removed_data['images'].append(img)
    elif img['id'] in other_images:
        kept_data['images'].append(img)

# Split annotations
for anno in data['annotations']:
    if anno['image_id'] in removed_images:
        removed_data['annotations'].append(anno)
    elif anno['image_id'] in other_images:
        kept_data['annotations'].append(anno)

# Split categories
# for cat in data['categories']:
#     if cat['id'] in remove_category_ids:
#         removed_data['categories'].append(cat)
#     else:
#         kept_data['categories'].append(cat)

# Save the two datasets
removed_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-only-curry.json'
kept_path = '/mnt/nas/data/growth_zero/kids_sauce_food/kids-mask-removed-curry.json'

with open(removed_path, 'w+', encoding='utf-8') as f:
    json.dump(removed_data, f, indent=2, ensure_ascii=False)

# with open(kept_path, 'w+', encoding='utf-8') as f:
#     json.dump(kept_data, f, indent=2, ensure_ascii=False)

print(f"Removed categories dataset saved to: {removed_path}")
print(f"Kept categories dataset saved to: {kept_path}")
print(f"Number of images in removed dataset: {len(removed_data['images'])}")
print(f"Number of images in kept dataset: {len(kept_data['images'])}")

print(f"Number of annotations in removed dataset: {len(removed_data['annotations'])}")
print(f"Number of annotations in kept dataset: {len(kept_data['annotations'])}")