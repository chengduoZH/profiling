from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import time
import sys, os

with_gpu = os.getenv('WITH_GPU', '0') != '0'

start_time = time.time()

def convolution_net(data, label, input_dim, class_dim=2, emb_dim=128,
                    hid_dim=128):
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim])
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3)
        #act="tanh",
        #pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4)
        #act="tanh",
        #pool_type="sqrt")
    prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                 size=class_dim,
                                 act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.002)
    adam_optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)
    return avg_cost, accuracy, accuracy.metrics[0]


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def main():
    BATCH_SIZE = 100
    PASS_NUM = 10
    global start_time
    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost, accuracy, acc_out = convolution_net(
        data, label, input_dim=dict_dim, class_dim=class_dim)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=BATCH_SIZE*10),
        batch_size=BATCH_SIZE)

    place = fluid.GPUPlace(0) if with_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())
    
    for pass_id in xrange(PASS_NUM):
        accuracy.reset(exe)
        iterator = 0
        pass_start_time = time.time()
        for data in train_data():
            cost_val, acc_val = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[cost, acc_out])
            pass_acc = accuracy.eval(exe)
            
            if iterator % 100 == 0:
                print("pass_id="+str(pass_id) + "  batch_id="+str(iterator) + " cost=" + str(cost_val) + " acc=" + str(acc_val) +
                  " pass_acc=" + str(pass_acc))
            iterator += 1
        print("pass_id=" + str(pass_id) + "  time consume:" + str(time.time() - pass_start_time))


if __name__ == '__main__':
    main()
