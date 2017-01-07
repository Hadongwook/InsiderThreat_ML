import tensorflow as tf
import numpy as np

char_rdic = list('helo') # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id
print(char_rdic)
print(char_dic)

x_data = np.array([
    [1,0,0,0],      #h
    [0,1,0,0],      #e
    [0,0,1,0],      #l
    [0,0,1,0]       #l
], dtype='f')
print(x_data)
sample = [char_dic[c] for c in 'hello'] #to index
print("sample: ", sample)
print(type(sample))
print(type(sample[0]))

char_vocab_size = len(char_dic)
rnn_size = char_vocab_size
time_step_size = 4
batch_size = 1

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

logit = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
print("!", sample[1:])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logit], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print("Xsplit: ", sess.run(X_split))
    print("outputs: ", sess.run(outputs))
    print("state: ", sess.run(state))
    print("logit: ", sess.run(logit))
    print("targets: ", sess.run(targets))
    print("weights: ", sess.run(weights))

    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logit, 1))
        print("%r, %r" %(result, [char_rdic[t] for t in result]))
    test = np.array([
        [1, 0, 0, 0],  # h
        [0, 1, 0, 0],  # e
        [0, 1, 0, 0],  # l
        [0, 0, 1, 0]  # l
    ], dtype='f')
    T_split = tf.split(0, time_step_size, test)
    outputs, state = tf.nn.rnn(rnn_cell,T_split, state)
    tresult = sess.run(tf.arg_max(logit, 1))
    print("%r, %r" % (result, [char_rdic[t] for t in tresult]))
