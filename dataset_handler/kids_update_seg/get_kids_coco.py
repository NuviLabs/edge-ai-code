from nuviAPI.hub import hub_api

dataset = hub_api.get_dataset("mask:25.01.02", folder='/mnt/nas/data/.hubAPI', download=True)
print(len(dataset.dataset['images']))
# dataset = hub_api.get_dataset(name=data_name, type=target_type, download=True, folder=root)

# from glob import glob

# dataset = glob('/mnt/nas/data/.hubAPI/ediya/**/*.json', recursive=True)
# print('')