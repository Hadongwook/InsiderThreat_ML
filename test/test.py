import numpy as np

a = np.arange(100)
print(a)
for i in range(0, len(a), 7):
    temp = a[i:i+7]
    if len(temp) < 7:
        temp = np.pad(temp, (0,7-len(temp)), 'edge')

    print(temp)



