import numpy as np

def set_xdata(sequence):
    input = np.array([0, 0, 0, 0, 0, 0, 0], dtype='f')
    for i in sequence:
        if i == 0:
            temp = np.array([0, 0, 0, 0, 0, 0, 1], dtype='f')
            input = np.vstack([input,temp])
        elif i == 1:
            temp = np.array([0, 0, 0, 0, 0, 1, 0], dtype='f')
            input = np.vstack([input, temp])
        elif i == 2:
            temp = np.array([0, 0, 0, 0, 1, 0, 0], dtype='f')
            input = np.vstack([input, temp])
        elif i == 3:
            temp = np.array([0, 0, 0, 1, 0, 0, 0], dtype='f')
            input = np.vstack([input, temp])
        elif i == 4:
            temp = np.array([0, 0, 1, 0, 0, 0, 0], dtype='f')
            input = np.vstack([input, temp])
        elif i == 5:
            temp = np.array([0, 1, 0, 0, 0, 0, 0], dtype='f')
            input = np.vstack([input, temp])
        else:
            temp = np.array([1, 0, 0, 0, 0, 0, 0], dtype='f')
            input = np.vstack([input, temp])
    input = np.delete(input, 0, 0)
    return input