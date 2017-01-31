import pandas as pd
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u87.csv', sep=',')
cols = user.columns.tolist()
cols = cols[3:-1] + cols[-1:]
user = np.array(user[cols])

train_set = user[0:60]
test_set = user[0:-1]
n_size = 7

def get_patterns(train_set):
    for i in range(0,len(train_set)):
        sequence = train_set[i]
        sequence = np.array(sequence[~np.isnan(sequence)])
        if len(sequence) < n_size:
            print('extend sequence')
            sequence = np.lib.pad(sequence, (0, (n_size - len(sequence))), 'edge')
        print(sequence)

        for j in range(0,(len(sequence)-n_size)):
            if i == 0 and j == 0:
                patterns = [np.array(sequence[j:(j+n_size)])]
            else:
                patterns = np.append(patterns, [sequence[j:(j+n_size)]], axis=0)
        patterns = np.vstack({tuple(row) for row in patterns})
        print("sequence ",i,"patterns:", patterns)
        print(len(patterns))
    return patterns

test_patterns = get_patterns(train_set)

seq_length = n_size
batch_size = 40
vocab_size = 7
embedding_dim = 50
memory_dim = 100

enc_inp = [tf.placeholder(tf.int32, shape = (None,), name="inp%i" % t) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape = (None,), name="labels%i" % t) for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype = tf.float32) for labels_t in labels]
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32)] + enc_inp[:-1])
prev_mem = tf.zeros((batch_size, memory_dim))


cell = tf.nn.rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_size = 100 )

loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)


def train_seq():
    X = test_patterns
    Y = X[:]

    X = np.array(X, dtype=int).T
    Y = np.array(Y, dtype=int).T
    # print(X.shape)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
        # print(feed_dict)

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


sess.run(tf.global_variables_initializer())
for t in range(5001):
    loss_t = train_seq()
    if t % 100 == 0:
        print("step: ", t, "loss: ", loss_t)


