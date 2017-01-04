import numpy as np
import pandas as pd

log = pd.read_csv('C:/Users/lab/Desktop/lab/InsiderThreat/CERT dataset/r4.2.tar/r4.2/r4.2/logon.csv', sep=',')

users = log['user'].unique()

no = 0
dic = pd.DataFrame(users, index=range(0,len(users)),columns=['user'])

dic.to_csv('C:/Users/lab/InsiderThreat/dictionary.csv', sep=',')
