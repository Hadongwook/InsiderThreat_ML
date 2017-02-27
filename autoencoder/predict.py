import pandas as pd
import numpy as np

anomalies = np.array([['name', 'date', 'loss']])
dict = pd.read_csv('C:/Users/lab/InsiderThreat/dictionary.csv', sep=',')
dict = dict['user']
for n in range(1000):
    file = 'u' + str(n) + '.csv'
    loss = pd.read_csv('C:/Users/lab/InsiderThreat/autoenc2/seq4/autoenc_w4_' + file, sep=',')
    date = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_' + file, sep=',')

    loss = loss['0']
    date = date['date']
    loss = np.delete(np.array(loss), 0, 0)

    for i in range(len(loss)):
        if loss[i] >= 0.3:
            anomalies = np.append(anomalies, np.array([[dict[n], date[i], loss[i]]]), axis=0)
    print("user "+str(n)+" done.")



anomalies = np.delete(anomalies, 0, 0)
anomalies = pd.DataFrame(anomalies, columns=['user', 'date', 'loss'])
anomalies.to_csv('C:/Users/lab/InsiderThreat/autoenc2/predict/w4_predict0.3_ver2.csv', sep=',')
