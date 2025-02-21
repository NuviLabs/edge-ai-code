from nuviAPI.hub import hub_api
from nuviAPI.hub.db.data import FilterQuery

hub_api.make_dataset(
    name="ediya:25.01.06",
    creator="kaeun.kim",
    filters = FilterQuery(has_gt= True, has_depth= False, sources= ["ediya"], ignore_sources= ["military-1001", "highschool-4001", "samsung-welstory", "welstory", "middleschool-3001", "nuvilab", "nuvi-expo", "alexandra-hospital", "nuvilabs", "microsoft-hq", "busan-aws"], ignore_sources_table= True, shape_type= ["mask"] ),
    description="Ediya gt dataset",
)

