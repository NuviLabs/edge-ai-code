import pandas as pd

df = pd.read_csv('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/merge_tests/[냠냠플레이] 잔반제로 모델 QA_image - QA.csv')
before_urls = df['before_url'].tolist()

candidate1 = pd.read_csv('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/merge_tests/[냠냠플레이] 잔반제로 모델 QA_image - archieved4(QA_범용).csv')
candidate2 = pd.read_csv('/home/kaeun.kim/kaeun-dev/nuvilab/dataset_handler/kids_sauce/merge_tests/[냠냠플레이] 잔반제로 모델 QA_image - archieved4(QA_덮밥only).csv')
for before_url in before_urls:
    candidate1_url = candidate1[candidate1['s3_uri_before'] == before_url]
    candidate2_url = candidate2[candidate2['before_url'] == before_url]

    try:
        candidate2_url['Unnamed: 4']
    except:
        print("not found in candidate2")
   

    print()
print(df.head())