import numpy as np

a = [1,2,3,4,5]
sequence = [0,6]
n_size = 7
print(np.lib.pad(a, (0,3),'edge'))
temp = np.lib.pad(sequence,(0,(n_size - len(sequence))), 'edge')
print(temp)
