import pandas as pd

user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u989.csv', sep=',')
train_sdate = pd.to_datetime('2010-01-01')
train_edate = pd.to_datetime('2010-04-30')
test_sdate = pd.to_datetime('2010-05-01')
test_edate = pd.to_datetime('2011-05-31')


#길이를 맞추기 위해 마지막 행위로 채움
user = user.fillna(method='ffill', axis=1)
user = pd.DataFrame(user)
user.to_csv("test/test.csv", sep=',')

#트레이닝 셋을 만들기 위해 1월 1일 부터 2월 28일 까지의 행위만을 추출
train_set = user.loc[(train_sdate <= pd.to_datetime(user['date'])) & (pd.to_datetime(user['date']) < train_edate)]
train_set = pd.DataFrame(train_set)

#트레이닝 세트에 결과와 bias 값을 셋팅 1월 1일 부터 2월 28일 까지의 행위를 정상행위로 판단
cols = train_set.columns.tolist()
cols = cols[2:-1]
train_set = train_set[cols]
train_set['anomal'] = 0
train_set['nomal'] = 1
train_set['bias'] = 1
cols = train_set.columns.tolist()
cols = cols[-1:] + cols[:-1]
train_set = train_set[cols]
train_set.to_csv("test/trainset.csv", sep=",")

#테스트 셋을 위해 동일한 작업 수행 대신 결과값은 설정하지 않는다.
test_set = user.loc[(test_sdate <= pd.to_datetime(user['date'])) & (pd.to_datetime(user['date']) < test_edate)]
test_set = pd.DataFrame(test_set)
cols = test_set.columns.tolist()
cols = cols[2:-1]
test_set = test_set[cols]
test_set['bias'] = 1
cols = test_set.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_set = test_set[cols]
test_set.to_csv("test/testset.csv", sep=",")
