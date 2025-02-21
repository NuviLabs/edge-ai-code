from nuviAPI.hub import hub_api
import json

dataset = hub_api.get_dataset("kids-mask:25.01.10")

path = '/mnt/nas/data/.hubAPI/kids-mask:25.01.10.json'

with open(path, 'w+') as f:
    json.dump(dataset.dataset, f)

print(len(dataset.dataset['images']))