import pandas as pd
import numpy as np

user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u87.csv', sep=',')
cols = user.columns.tolist()
cols = cols[3:-1] + cols[-1:]
user = np.array(user[cols])

train_set = user[0:60]
n_size = 7

def get_train_patterns(train_set):
    for i in range(0,len(train_set)):
        sequence = train_set[i]
        sequence = np.array(sequence[~np.isnan(sequence)])

        for j in range(0,(len(sequence)-n_size)):
            if i == 0 and j == 0:
                patterns = [np.array(sequence[j:(j+n_size)])]
            else:
                patterns = np.append(patterns, [sequence[j:(j+n_size)]], axis=0)
        patterns = np.vstack({tuple(row) for row in patterns})
        print("sequence ",i,"patterns:", patterns)
        print(len(patterns))
    return patterns

patterns = get_train_patterns(train_set)


