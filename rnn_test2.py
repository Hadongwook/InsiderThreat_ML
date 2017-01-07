import numpy as np
import pandas as pd
import tensorflow as tf
import dataprocess as dp  #데이터 처리를 위한 함수를 담은 파이썬 파일

char_rdic = list('0123456')
char_dic = {w: i for i, w in enumerate(char_rdic)}

# 트레이닝 세트를 만듬 1월부터 2월까지의 행위
user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u989.csv', sep=',')
user = user.fillna(method='ffill', axis=1, downcast={'6.0': '6'})
train_set = pd.DataFrame(user[0:40])

cols = train_set.columns.tolist()
cols = cols[3:-1] + cols[-1:]
train_set = train_set[cols]

train_set = np.array(train_set, dtype='f')
x_data = np.empty([1, 54, 7], dtype='f')
sample = np.empty([1, 54], dtype=int)
for seq in train_set:
    temp_x = dp.set_xdata(seq)    #make 2d numpy array
    temp_y = [int(i) for i in seq]
    x_data = np.vstack((x_data, temp_x[None]))
    sample = np.vstack((sample, temp_y[0:]))
x_data = np.delete(x_data, 0, 0)
sample = np.delete(sample, 0, 0)
print("X: ", x_data)
print("Y: ", sample)
#sequence = train_set[0]
#x_data = dp.set_xdata(sequence)
#sample = [int(i) for i in sequence]

X = tf.placeholder('float', [None, 54, 7])
Y = tf.placeholder('int64', [None, 54])

output_size = len(char_dic)
rnn_size = output_size
time_step_size = 54
batch_size = 1

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

logit = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[0:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logit], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        sess.run(train_op)
        if i % 20 == 0:
            result = sess.run(tf.arg_max(logit, 1), feed_dict={X: x_data, Y: sample})
            #print('%r, %r' %(result, [char_rdic[t] for t in result]))
        print('%r, %r' % (result, [char_rdic[t] for t in result]))

