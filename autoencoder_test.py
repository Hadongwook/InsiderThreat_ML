import numpy as np
import tensorflow as tf
import pandas as pd

sess = tf.Session()

user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u989.csv', sep=',')
user = user.fillna(method='ffill', axis=1, downcast={'6.0': '6'})
train_set = pd.DataFrame(user[0:40])

cols = train_set.columns.tolist()
cols = cols[3:-1] + cols[-1:]
train_set = train_set[cols]



seq_length = 5
print(seq_length)
batch_size = 64
vocab_size = 7
embedding_dim = 50
memory_dim = 100

enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" %t) for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name = "labels%i" % t) for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")] + enc_inp[:-1])
prev_mem = tf.zeros((batch_size, memory_dim))

cell = tf.nn.rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_size=50)

loss = tf.nn.seq2seq.sequence_loss(dec_outputs,labels,weights,vocab_size)
tf.scalar_summary("loss", loss)

magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.scalar_summary("magnitude at t=1", magnitude)

summary_op = tf.merge_all_summaries()

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())

def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(batch_size)]
    Y = X[:]
    #print(np.array(X).T.shape)
    X = np.array(X).T
    Y = np.array(Y).T

    #print(X)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
    #print("feed: ", feed_dict )
    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

for t in range(1000):
    loss_t, summary = train_batch(batch_size)
    if t % 50 == 0:
        print("loss_t: ", loss_t)
X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(1)]
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)


print("Xbatch: ", X_batch.shape)
print(np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T)