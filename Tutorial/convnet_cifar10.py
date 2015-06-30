#coding: utf-8
import os
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

def relu(x):
    """Rectified Linear Unit"""
    return theano.tensor.switch(x < 0, 0, x)

class ConvLayer(object):
    """畳み込みニューラルネットの畳み込み層"""
    def __init__(self, rng, input, image_shape, filter_shape):
        # 入力の特徴マップ数は一致する必要がある
        assert image_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX),  # @UndefinedVariable
            borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)  # @UndefinedVariable
        self.b = theano.shared(value=b_values, borrow=T)

        # 入力の特徴マップとフィルタの畳み込み
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)

        # バイアスを加える
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class PoolingLayer(object):
    """畳み込みニューラルネットのプーリング層
    この実装ではプーリング層にパラメータはない"""
    def __init__(self, rng, input, poolsize=(2, 2)):
        # Max-poolingを用いて各特徴マップをダウンサンプリング
        pooled_out = downsample.max_pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True)

        self.output = pooled_out

def unpickle(filename):
    import cPickle
    fp = open(filename, "rb")
    d = cPickle.load(fp)
    fp.close()
    return d

def load_data(cifar_dir):
    train_set = [[], []]
    valid_set = [[], []]
    test_set =  [[], []]

    # 訓練セットをロード
    for i in range(1, 5):
        d = unpickle(os.path.join(cifar_dir, "data_batch_%d" % i))
        train_set[0].extend(d["data"])
        train_set[1].extend(d["labels"])
        #print d["data"].reshape((10000, 3, 32, 32)).shape

    # バリデーションセットをロード
    d = unpickle(os.path.join(cifar_dir, "data_batch_5"))
    valid_set[0].extend(d["data"])
    valid_set[1].extend(d["labels"])

    # テストセットをロード
    d = unpickle(os.path.join(cifar_dir, "test_batch"))
    test_set[0].extend(d["data"])
    test_set[1].extend(d["labels"])

    # 画像の可視化
    import matplotlib.pyplot as plt
    image = np.array(train_set[0][0].reshape(3, 32, 32))
    image = image.transpose((1, 2, 0))
    print image.shape
    plt.imshow(image)
    plt.show()
    exit()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    print type(train_set_x)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def evaluate_cifar10(learning_rate=0.1, n_epochs=200, batch_size=500):
    rng = np.random.RandomState(23455)

    # 学習データのロード
    datasets = load_data("cifar-10-batches-py")

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # ミニバッチの数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "building the model ..."

    # ミニバッチのインデックスを表すシンボル
    index = T.lscalar()

    # ミニバッチの学習データとラベルを表すシンボル
    x = T.matrix('x')
    y = T.ivector('y')

    # 入力
    # 入力のサイズを4Dテンソルに変換
    # batch_sizeは訓練画像の枚数
    # チャンネル数は1
    # (28, 28)はMNISTの画像サイズ
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    layer0 = ConvLayer(rng,
                input=layer0_input,
                image_shape=(batch_size, 1, 28, 28),
                filter_shape=(20, 1, 5, 5))

    layer1 = PoolingLayer(rng,
                          input=layer0.output,
                          poolsize=(2, 2))

    layer2 = ConvLayer(rng,
                       input=layer1.output,
                       image_shape=(batch_size, 20, 12, 12),
                       filter_shape=(50, 20, 5, 5))

    layer3 = PoolingLayer(rng,
                          input=layer2.output,
                          poolsize=(2, 2))

    # 隠れ層への入力
    layer4_input = layer3.output.flatten(2)

    # 全結合された隠れ層
    layer4 = HiddenLayer(rng,
        input=layer4_input,
        n_in=50 * 4 * 4,
        n_out=500,
        activation=T.tanh)

    # 最終的な数字分類を行うsoftmax層
    layer5 = LogisticRegression(input=layer4.output, n_in=500, n_out=10)

    # コスト関数を計算するシンボル
    cost = layer5.negative_log_likelihood(y)

    # index番目のテスト用ミニバッチを入力してエラー率を返す関数を定義
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # index番目のバリデーション用ミニバッチを入力してエラー率を返す関数を定義
    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        })

    # パラメータ
    params = layer5.params + layer4.params + layer2.params + layer0.params

    # コスト関数の微分
    grads = T.grad(cost, params)

    # パラメータ更新式
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    # index番目の訓練バッチを入力し、パラメータを更新する関数を定義
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })

    print "train model ..."

    # eary-stoppingのパラメータ
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    fp1 = open("validation_error.txt", "w")
    fp2 = open("test_error.txt", "w")

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print "epoch %i, minibatch %i/%i, validation error %f %%" % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100)
                fp1.write("%d\t%f\n" % (epoch, this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        # 十分改善したならまだ改善の余地があるためpatienceを上げてより多くループを回せるようにする
                        patience = max(patience, iter * patience_increase)
                        print "*** iter %d / patience %d" % (iter, patience)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # テストデータのエラー率も計算
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print "    epoch %i, minibatch %i/%i, test error of best model %f %%" % (epoch, minibatch_index + 1, n_train_batches, test_score * 100)
                    fp2.write("%d\t%f\n" % (epoch, test_score * 100))

            # patienceを超えたらループを終了
            if patience <= iter:
                done_looping = True
                break

    fp1.close()
    fp2.close()

    end_time = time.clock()
    print "Optimization complete."
    print "Best validation score of %f %% obtained at iteration %i, with test performance %f %%" % (best_validation_loss * 100.0, best_iter + 1, test_score * 100.0)
    print "The code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((end_time - start_time) / 60.0)

if __name__ == '__main__':
    evaluate_cifar10()
