# from nuviAPI.hub import hub_api
# from nuviAPI.hub.db.db_handler import FilterQuery


# filter = FilterQuery(
#     has_gt=True,
#     sources=['carleton-university'],
#     start_date='241111',
#     end_date='241118',
#     ba='A',
# )

# dataset = hub_api.make_dataset(name='carleton-university-gt:1.0', creator='Kaeun Kim', filters=filter)



from nuviAPI.data import data_api
import json

tmp = data_api.labelme_to_coco('/mnt/nas/data/kaeun/q4/carleton-university/annotations/')
# save to json
with open('/mnt/nas/data/kaeun/q4/carleton-university/coco.json', 'w+') as f:
    json.dump(tmp, f)

