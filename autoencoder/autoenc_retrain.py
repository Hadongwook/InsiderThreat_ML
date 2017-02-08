import pandas as pd
import numpy as np
import tensorflow as tf


# train set에서 나올 수 있는 모든 패턴을 찾아낸다
def get_patterns(data_set, size):
    # print(len(data_set))
    patterns = np.random.rand(1, size)
    for i in range(0, len(data_set)):
        sequence = data_set[i]
        sequence = np.array(sequence[~np.isnan(sequence)])

        if len(sequence) <= size:
            print('extend sequence')
            sequence = np.lib.pad(sequence, (0, (size - len(sequence))), 'edge')
            patterns = np.append(patterns, [sequence], axis=0)

        for j in range(0, (len(sequence) - size)):
            patterns = np.append(patterns, [sequence[j:(j + size)]], axis=0)
    # 초기화 할때 사용한 랜덤값 제거 와 중복 패턴 제거
    patterns = np.delete(patterns, 0, 0)
    patterns = np.vstack({tuple(row) for row in patterns})
    # print(len(patterns))
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

# 하루 일과에서 찾아낼 패턴의 길이
n_size = 4

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

    saver = tf.train.Saver()
    for n in range(1000):
        #tensorflow session open
        #sess = tf.InteractiveSession()
        #유저 파일 하나씩 가져와서 실행
        file = 'daily_f_u' + str(n) + '.csv'
        user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/' + file, sep=',')

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
                loss_t = train_seq(train_patterns)
                if t == 0:
                    print("training...")
                    """
                    if t % 100 == 0:
                        print("step: ", t, "loss: ", loss_t)
                    """
            print("user ", n, "trained. loss: ", loss_t)
            #학습내용 저장
            save_path = saver.save(sess, 'C:/Users/lab/InsiderThreat/autoenc2/saver/saver_' + file + '.ckpt')

        with tf.Session() as sess:
            loss_arr = np.zeros((60, 1))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'C:/Users/lab/InsiderThreat/autoenc2/saver/saver_' + file + '.ckpt')

            for x in range(60, len(test_set)):
                if x % 5 == 0:
                    train_patterns = [[1, 1, 1, 1]]
                seq_len = len(np.array(test_set[x][~np.isnan(test_set[x])]))
                if seq_len > 3:
                    seq = [test_set[x]]
                    test_patterns = get_patterns(seq, n_size)
                    X = test_patterns
                    X = np.array(X, dtype=int).T
                    Y = X
                    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
                    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
                    dec_outputs_batch = sess.run(dec_outputs, feed_dict)
                    ploss = sess.run(loss, feed_dict)
                else:
                    ploss = 0

                loss_arr = np.append(loss_arr, [[ploss]], axis=0)
                print(x, ", ", ploss)

                if (x + 1) % 5 == 0:
                    temp = user[(x + 1) - 60:(x + 1)]
                    temp2 = np.where(loss_arr[(x + 1) - 60:(x + 1)] < 0.3)[0]
                    train_set = np.stack({tuple(temp[n]) for n in temp2})
                    print(len(train_set))

                    train_patterns = get_patterns(train_set, n_size)
                    print("week", (x + 1) / 5, " training")
                    for t in range(5001):
                        loss_t = train_seq(train_patterns)
                    save_path = saver.save(sess, 'C:/Users/lab/InsiderThreat/autoenc2/saver/saver_' + file + '.ckpt')

        pd.DataFrame(loss_arr).to_csv("C:/Users/lab/InsiderThreat/autoenc2/retrain/retrain_w4_u"+str(n)+".csv", sep=",")
        print("user ",n," done.")