import pandas as pd
from sklearn.cluster import KMeans

insiders = pd.read_csv('C:/Users/lab/InsiderThreat/insiders.csv', sep=',')
insiders = insiders.user
dic = pd.read_csv('C:/Users/lab/InsiderThreat/dictionary.csv', sep=',')
dic = pd.DataFrame(dic)
threat_to_file = pd.DataFrame()
for insider in insiders:
    user_num = dic[dic['user'] == insider].index.tolist()[0]
    file = 'daily_f_u'+str(user_num)+'.csv'
    print(file)
    #악의적 행동을 한 직원의 데일리 워크 파일 로드한 후 데이터 길이를 맞춤
    user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/'+file, sep=',')
    user = pd.DataFrame(user)
    user_t = user.fillna(method='ffill', axis=1)
    #print(user_t)
    #행동만 추출
    cols = user_t.columns.tolist()
    cols = cols[2:-1] + cols[-1:]
    user_t = user_t[cols]
    #Kmeans 로 클러스터링 3개로 분류
    kmeans_model = KMeans(n_clusters=3)
    kmeans = kmeans_model.fit(user_t)
    labels = kmeans_model.labels_
    #분류된 라벨을 각 날마다 붙임
    user['label'] = labels

    user.to_csv('C:/Users/lab/InsiderThreat/Kmeans/km_'+file)

    """
    #라벨이 가장  조금 나온 행위가 악의적 행위로 판단 이것을 따로 저장
    label = user['label'].value_counts().index[2]
    threat = pd.DataFrame([[insider, user.loc[user['label'] == label]['date']]])
    threat_to_file = pd.concat([threat_to_file, threat])
    print(kmeans.cluster_centers_)
    break
threat_to_file.to_csv("C:/Users/lab/InsiderThreat/Kmeans/Kmens_threat.csv", sep=',')
"""
"""
print(labels)
print(kmeans.cluster_centers_)
#print(kmeans.predict(normal), kmeans.predict(threat))

"""