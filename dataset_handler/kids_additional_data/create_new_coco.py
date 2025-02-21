from nuviAPI.hub import hub_api
from nuviAPI.hub.db.data import FilterQuery

hub_api.make_dataset(
    name="mask:25.01.02",
    creator="kaeun.kim",
    filters = FilterQuery(has_gt= True, has_depth= False, sources= ["baeksong-dc", "kwiil-ms", "kmoon-es", "kunkook-university-ms", "jeonju-hyorim-es", "jeonju-seomun-es", "saengmyeongsup-dc", "inhwa-es", "ogeum-es", "saewoogaehana-dc", "anchung-ms", "shinlim-ms", "sawoo-es", "sentber-dc", "ediya", "yongdang-dc", "pinocchio-dc", "oryukdo-dc", "myeongin-dc", "seoho-es", "bugang-es", "pyoseon-ms", "chunghyun-ms", "manseong-ms", "suseong-ms", "daejeong-w-hs", "sehwa-ms", "yongho-hs", "songho-hs", "snfl-hs", "buram-hs", "kongrung-ms", "hwarang-es", "wonheung-es", "naju-es", "gongsan-es", "gwangyang-hs", "haean-es", "saewoomteo-dc", "yanggok-hs", "sangroksu-dc", "seoul-girls-commercial-hs", "siripjanghyunyedaum-dc", "jeonju-seoil-es", "sunhye-dc", "gangjeong-es", "yangjae2dong-dc", "yongho-dc", "yongsan-dc", "hwbong-es", "yongsanguchungjikjang-dc", "samgye-hs", "gwanghui-hs", "bitgaon-dc", "changsin-dc", "cheonsa-dc", "choonghyeon-dc", "daeyeon-dc", "dodamdodam-dc", "eunhaengtree-dc", "gurip-suyu1dong-dc", "dongbingo-dc", "durumi-dc", "gamman-dc", "gurip-guil-dc", "gwangmyeong-gurumsan-dc", "haannuri-dc", "hanti-dc", "he-mang-dc", "kyungsung-dc", "malgeunsoop-dc", "myeongryun-dc", "purunsoop-dc", "sejongmaeul-dc", "gurip-daejo-dc", "samyuk-ms", "sunam-k"], ignore_sources= ["military-1001", "highschool-4001", "samsung-welstory", "welstory", "middleschool-3001", "nuvilab", "nuvi-expo", "alexandra-hospital", "nuvilabs", "microsoft-hq", "busan-aws"], ignore_sources_table= True, shape_type= ["mask"] ),
    description="Mask dataset for 25.01.02",
)

