import torch
import json
import pandas as pd

extras = {'classes': '', 'dict': ''}
model_path = "/home/kaeun.kim/kaeun-dev/clip_image_encoder.pt"
model = torch.jit.load(f"{model_path}", _extra_files=extras)

classes = json.loads(extras['classes'])
food_class = pd.DataFrame(pd.read_csv("/home/kaeun.kim/kaeun-dev/food_classes.csv"))
nuvi_suop = food_class[food_class['형태 대분류']=='국류']['음식명'].tolist()
# Common Korean terms for soups and similar dishes
soup_terms_kr = ['탕', '국', '찌개', '스튜', '죽']

# Filtering for classes related to soup in both English and Korean
soup_related_classes = [cls.strip() for cls in classes if any(term in cls.lower() for term in soup_terms_kr)]
all_soup = list(set(soup_related_classes + nuvi_suop))
# save all_soup in a csv file
pd.DataFrame(all_soup).to_csv('all_soup.csv', index=False)
print('')