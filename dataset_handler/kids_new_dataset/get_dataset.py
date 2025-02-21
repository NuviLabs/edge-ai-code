import json
from nuviAPI.hub import hub_api

dataset = hub_api.get_dataset("kids-mask:25.02.19", folder='/mnt/nas/data/.hubAPI', download=True)
print(len(dataset.dataset['images']))

