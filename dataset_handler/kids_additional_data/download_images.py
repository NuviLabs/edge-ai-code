from nuviAPI import s3_api 
import os
dep2date = [
("gukbangdaehakyo-dc",241211),
("gukbangdaehakyo-dc",241210),
("gukbangdaehakyo-dc",241212),
("donghwanala-dc",241210),
("eunseon-yenung-dc",241219),
("sejongmaeul-dc",241210),
("gaon-dc",241210),
("gaon-dc",241212),
("gaon-dc",241211),
("byeolha-pradium-dc",241210),
("yeneung-easternkids-dc",241211),
("icheon-welfare-dc",241210),
("ihwa-dc",241210),
("nonsan-woori-dc",241210),
("seochogurip-haeun-dc",241213),
("ocheon-bodeumi-nanumi-dc",241210),
("saesomang-dc",241211),
("saesomang-dc",241212),
("saesomang-dc",241213),
("gongdan-suandeul-dc",241210),
("saeteo-dc",241210),
("saeteo-dc",241211),
("eokkaedongmu-dc",241210),
("purunsoop-dc",241211),
("parannara-dc",241211),
("parannara-dc",241212),
("jeil-dc",241127),
("areumsol-dc",241125),
("sangok-1dong-hana-dc",241126),
("areumsol-dc",241127),
("sejongmaeul-dc",241211),
("sejongmaeul-dc",241213),
("sangok-1dong-hana-dc",241129),
("icheon-welfare-dc",241213),
("ocheon-bodeumi-nanumi-dc",241213),
("ocheon-bodeumi-nanumi-dc",241212),
("jeil-dc",241129),
("sejongmaeul-dc",241212),
("jeil-dc",241128),
("byeolha-pradium-dc",241213),
("byeolha-pradium-dc",241211),
("byeolha-pradium-dc",241212),
("sangok-1dong-hana-dc",241127),
("yeneung-easternkids-dc",241213),
("yeneung-easternkids-dc",241212),
("icheon-welfare-dc",241212),
("ihwa-dc",241213),
("ihwa-dc",241212),
("nonsan-woori-dc",241213),
("nonsan-woori-dc",241212),
("areumsol-dc",241126),
("ocheon-bodeumi-nanumi-dc",241211),
("gongdan-suandeul-dc",241213),
("gongdan-suandeul-dc",241212),
("eunseon-yenung-dc",241129),
("eunseon-yenung-dc",241119),
("eunseon-yenung-dc",241112),
("eunseon-yenung-dc",241118),
("eunseon-yenung-dc",241111),
("eunseon-yenung-dc",241106),
("saeteo-dc",241226),
("saeteo-dc",241224),
("saeteo-dc",241219),
("saeteo-dc",241220),
("saeteo-dc",241217),
("eokkaedongmu-dc",241211),
("eokkaedongmu-dc",241212),
("eokkaedongmu-dc",241213),
("purunsoop-dc",241210),
("purunsoop-dc",241212),
("purunsoop-dc",241213),
("parannara-dc",241213),
("parannara-dc",241210),
("sejongmaeul-dc",240723),
("areumsol-dc",241122),
("saesomang-dc",241105),
("saesomang-dc",241106),
("saesomang-dc",241108),
("saesomang-dc",241107),
("saesomang-dc",241111),
("saesomang-dc",241114),
("saesomang-dc",241112),
("gaon-dc",241112),
("gaon-dc",241113),
("gaon-dc",241120),
("donghwanala-dc",241125),
("donghwanala-dc",241126),
("ocheon-bodeumi-nanumi-dc",241120),
("parannara-dc",241119),
("saeteo-dc",241218),
("saeteo-dc",241216),
("nonsan-woori-dc",241211),
("gongdan-suandeul-dc",241211),
("ihwa-dc",241211),
("icheon-welfare-dc",241211),
("parannara-dc",241121),
("parannara-dc",241122),
("parannara-dc",241118),
("ocheon-bodeumi-nanumi-dc",241118),
("donghwanala-dc",241129),
("donghwanala-dc",241127),
("gaon-dc",241122),
("gaon-dc",241121),
("gaon-dc",241119),
("gaon-dc",241118),
("gaon-dc",241115),
("gaon-dc",241114),
("gaon-dc",241111),
]

from_uris = []
to_uris = []
for dep, date in dep2date:
    date_string = str(date)
    # imgs = s3_api.list_objects('nuvi-data', f'{dep}/{date_string}/L/A', endswith=['png', '.json', 'jpg', 'jpeg', 'webp'])
    imgs = s3_api.list_objects('nuvi-data', f'{dep}/{date_string}/', endswith=['json'])
    for img in imgs:
        if 'inference' in img:
            continue
        from_uri = s3_api.make_uri('nuvi-data', img)
        from_uris.append(from_uri)
        img_info = from_uri.split('nuvi-data/')[-1]
        to_uri = os.path.join('/mnt/nas/data/.hubAPI/', img_info)
        to_uris.append(to_uri)
    # imgs = s3_api.list_objects('nuvi-data', f'{dep}/{date_string}/L/B', endswith=['.json'])
    # for img in imgs:
    #     if 'inference' in img:
    #         continue
    #     from_uri = s3_api.make_uri('nuvi-data', img)
    #     from_uris.append(from_uri)
    #     img_info = from_uri.split('nuvi-data/')[-1]
    #     to_uri = os.path.join('/mnt/nas/data/.hubAPI/', img_info)
    #     to_uris.append(to_uri)

s3_api.cp(from_uris, to_uris)