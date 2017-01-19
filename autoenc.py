import numpy as np
import pandas as pd
import tensorflow as tf

sess = tf.InteractiveSession()

insiders = pd.read_csv('C:/Users/lab/InsiderThreat/insiders.csv', sep=',')
insiders = insiders.user
dic = pd.read_csv('C:/Users/lab/InsiderThreat/dictionary.csv', sep=',')
dic = pd.DataFrame(dic)


for insider in insiders:
    user_num = dic[dic['user'] == insider].index.tolist()[0]
    file = 'daily_f_u'+str(user_num)+'.csv'
    print(file)
    #악의적 행동을 한 직원의 데일리 워크 파일 로드한 후 데이터 길이를 맞춤
    #user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/'+file, sep=',')
    user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u87.csv', sep=',')
    user = pd.DataFrame(user)
    user = user.fillna(method='ffill', axis=1)
    train_data = pd.DataFrame(user[0:40])

    cols = train_data.columns.tolist()
    cols = cols[3:-1] + cols[-1:]

    #트레이닝 데이터와 테스트 데이터
    train_data = np.array(train_data[cols], dtype=int)
    y_data = train_data
    test_data = pd.DataFrame(user)
    test_data = np.array(test_data[cols], dtype=int)


    seq_length = len(cols)
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
    dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_size = 100 )

    loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)


    def train_seq():
        X = train_data
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
    loss_t = 5
    t= 0
    while loss_t > 0.009:
        loss_t = train_seq()
        if t % 100 == 0:
            print("step: ", t, "loss: ", loss_t)
        t += 1

    #하루하루의 시퀀스를 입력하며 평균 확률 출력
    for x in range(len(test_data)):
        X = [test_data[x]]
        X = np.array(X, dtype=int).T
        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        dec_outputs_batch = sess.run(dec_outputs, feed_dict)
        result = np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T

        temp = [sess.run(tf.nn.softmax(dec_outputs_batch[t])) for t in range(seq_length)]

        p = [temp[i][0][X[i]] for i in range(len(temp))]
        print("day", x, ": ", np.array(p).mean())
    pd.DataFrame(p).to_csv('u12_autoenc.csv', sep=',')
    break