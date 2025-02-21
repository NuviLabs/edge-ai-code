from nuviAPI.hub import hub_api
from nuviAPI.hub.db.data import FilterQuery

hub_api.make_dataset(
    name="kids-mask:25.02.19",
    creator="kaeun.kim",
    filters = FilterQuery(
        has_gt= True, 
        has_depth= False, 
        # ignore_sources= ["military-1001", "highschool-4001", "samsung-welstory", "welstory", "middleschool-3001", "nuvilab", "nuvi-expo", "alexandra-hospital", "nuvilabs", "microsoft-hq", "busan-aws", "carleton-university"], 
        lastModified_start_date = "2025-02-19 00-00-00",






),
    description="Kids maskGT with updated annotation",
)
