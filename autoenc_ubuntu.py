# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf

# 하루 일과에서 찾아낼 패턴의 길이 7개의 행위로 이루어진 패턴으로 분석
n_size = 4

# train set에서 나올 수 있는 모든 패턴을 찾아낸다
def get_patterns(data_set, size):
    patterns = np.random.rand(1, 4)
    for i in range(0,len(data_set)):
        sequence = data_set[i]
        sequence = np.array(sequence[~np.isnan(sequence)])

        if len(sequence) <= size:
            print('extend sequence')
            sequence = np.lib.pad(sequence, (0, (size - len(sequence))), 'edge')
            patterns = np.append(patterns, [sequence], axis=0)

        for j in range(0,(len(sequence)-size)):
            patterns = np.append(patterns, [sequence[j:(j+size)]], axis=0)

    patterns = np.delete(patterns, 0, 0)
    patterns = np.vstack({tuple(row) for row in patterns})
    return patterns

def train_seq(patterns):
    X = patterns
    Y = X[:]

    X = np.array(X, dtype=int).T
    Y = np.array(Y, dtype=int).T
    # print(X.shape)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
    # print(feed_dict)

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


# RNN Autoencoder를 위한 변수 선언
seq_length = n_size
batch_size = 40
vocab_size = 7
embedding_dim = 50
memory_dim = 100
with tf.Graph().as_default():
    enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(seq_length)]
    labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(seq_length)]
    weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
    dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32)] + enc_inp[:-1])
    prev_mem = tf.zeros((batch_size, memory_dim))

    cell = tf.nn.rnn_cell.GRUCell(memory_dim)
    # with tf.variable_scope("myrnn") as scope:
    # if n > 492:
    # scope.reuse_variables()
    dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,
                                                                  embedding_size=100)

    loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)

    for n in range(0, 1000):
        #tensorflow session open
        #sess = tf.InteractiveSession()
        #유저 파일 하나씩 가져와서 실행
        file = 'daily_f_u' + str(n) + '.csv'
        user = pd.read_csv('/home/ubuntu/inputs/' + file, sep=',')

        date = user['date']

        print("start user ", n)
        #필요 없는 column 제거 후 60일간의 trainig set와 전체 일의 test set 구성.
        cols = user.columns.tolist()
        cols = cols[3:-1] + cols[-1:]
        user = np.array(user[cols])
        train_set = user[0:60]
        test_set = user
        train_patterns = get_patterns(train_set, n_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for t in range(5001):
                loss_t = train_seq()
                if t == 0:
                    print("training...")
                    """
                    if t % 100 == 0:
                        print("step: ", t, "loss: ", loss_t)
                    """
            print("user ", n, "trained. loss: ", loss_t)
            loss_arr = [[loss_t]]
            for x in range(len(test_set)):
                if x == 0:
                    print("testing...")
                seq = [test_set[x]]
                test_patterns = get_patterns(seq, n_size)
                X = test_patterns
                X = np.array(X, dtype=int).T
                Y = X

                feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
                feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
                dec_outputs_batch = sess.run(dec_outputs, feed_dict)
                ploss = sess.run(loss, feed_dict)
                #print("loss: ", ploss)
                #print("outpupts: ", np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T, "\nloss: ", ploss)
                loss_arr = np.append(loss_arr, [[ploss]], axis=0)
                print(date[x], ploss)

        pd.DataFrame(loss_arr).to_csv("/home/ubuntu/ouputs/win4/autoenc_w4_u"+str(n)+".csv", sep=",")
        print("user ",n," done.")