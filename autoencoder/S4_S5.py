import pandas as pd
import numpy as np

files =['scenario4.csv', 'scenario5.csv']

for file in files:
    user = pd.read_csv('C:/Users/lab/InsiderThreat/' + file, sep=',')
    rng = pd.date_range('2010-01-01', '2011-05-30')
    total = pd.DataFrame()

    for date in rng:
        temp = user.loc[(date <= pd.to_datetime(user['date'])) & (pd.to_datetime(user['date']) < date + 1)]
        temp = pd.DataFrame(temp)

        if not temp.empty:
            length = len(temp)

            # print(date,"is not empty")
            # 각 행위를 편의상 숫자로 표현
            temp.loc[temp.activity == 'Logon', 'activity'] = 0
            temp.loc[temp.activity == 'http', 'activity'] = 1
            temp.loc[temp.activity == 'email', 'activity'] = 2
            temp.loc[temp.activity == 'file', 'activity'] = 3
            temp.loc[temp.activity == 'Connect', 'activity'] = 4
            temp.loc[temp.activity == 'Disconnect', 'activity'] = 5
            temp.loc[temp.activity == 'Logoff', 'activity'] = 6

            # 하루의 작업 시퀀스를 나타내기 위한 처리 날짜, 시퀀스 길이, 시퀀스 순으로 정렬
            dayact = pd.DataFrame(data=temp['activity'].values, index=list(range(0, len(temp))))
            dayact = np.transpose(dayact)
            dayact['length'] = length
            cols = dayact.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            dayact = dayact[cols]
            dayact['date'] = date
            cols = dayact.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            dayact = dayact[cols]

            # print(dayact)
            # 시퀀스를 붙여나감
            total = pd.concat([total, dayact])

    total.to_csv("C:/Users/lab/InsiderThreat/daily_work/daily_f_" + file, sep=',')