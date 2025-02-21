from nuviAPI.hub import hub_api
import json

dataset = hub_api.get_dataset("carleton-university-gt:25.01.16", folder='/mnt/nas/data/.hubAPI', download=True)

print(len(dataset.dataset['images']))