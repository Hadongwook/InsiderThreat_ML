import numpy as np
import tensorflow as tf

import pandas as pd

sess = tf.Session()

user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u989.csv', sep=',')
user = user.fillna(method='ffill', axis=1, downcast={'6.0': '6'})
train_set = pd.DataFrame(user[0:40])

cols = train_set.columns.tolist()
cols = cols[3:-1] + cols[-1:]
x_data = np.array(train_set[cols], dtype=int)        #shape (40,54)
y_data = x_data

seq_length = len(cols)          # length = 54
#print(seq_length)
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
dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_size = 100)

loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)


def train_seq():
    X = x_data
    Y = X[:]

    X = np.array(X, dtype=int).T
    Y = np.array(Y, dtype=int).T
    #print(X.shape)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
    #print(feed_dict)

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for t in range(5000):
        loss_t = train_seq()
        if t % 100 == 0:
            print("step: ", t , "loss: ", loss_t)


    X_batch = [np.random.choice(vocab_size, size=(seq_length,)) for _ in range(1)]
    X_batch = np.array(X_batch).T

    X = x_data
    X = np.array(X, dtype=int).T
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    dec_outputs_batch = sess.run(dec_outputs, feed_dict)

    print("Xbatch: ", X)
    print(np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T)
