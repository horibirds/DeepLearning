#coding: utf-8
import numpy as np
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """畳み込みニューラルネットの畳み込み層＋プーリング層"""
    def __init__(self, rng, input, image_shape, filter_shape, poolsize=(2, 2)):
        pass


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz', batch_size=500):
    rng = np.random.RandomState(23455)

    # 学習データのロード
    datasets = load_data(dataset)
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

    # 最初の畳み込み層+プーリング層
    # 畳み込みに使用するフィルタサイズは5x5ピクセル
    # 畳み込みによって画像サイズは28x28ピクセルから24x24ピクセルに落ちる
    # プーリングによって画像サイズはさらに12x12ピクセルに落ちる
    # 特徴マップ数は20枚でそれぞれの特徴マップのサイズは12x12ピクセル
    # 最終的にこの層の出力のサイズは (batch_size, 20, 12, 12) になる
    layer0 = LeNetConvPoolLayer(rng,
                input=layer0_input,
                image_shape=(batch_size, 1, 28, 28),  # 入力画像のサイズを4Dテンソルで指定
                filter_shape=(20, 1, 5, 5),           # フィルタのサイズを4Dテンソルで指定
                poolsize=(2, 2))

    # layer0の出力がlayer1への入力となる
    # layer0の出力画像のサイズは (batch_size, 20, 12, 12)
    # 12x12ピクセルの画像が特徴マップ数分（20枚）ある
    # 畳み込みによって画像サイズは12x12ピクセルから8x8ピクセルに落ちる
    # プーリングによって画像サイズはさらに4x4ピクセルに落ちる
    # 特徴マップ数は50枚でそれぞれの特徴マップのサイズは4x4ピクセル
    # 最終的にこの層の出力のサイズは (batch_size, 50, 4, 4) になる
    layer1 = LeNetConvPoolLayer(rng,
                input=layer0.output,
                image_shape=(batch_size, 20, 12, 12), # 入力画像のサイズを4Dテンソルで指定
                filter_shape=(50, 20, 5, 5),          # フィルタのサイズを4Dテンソルで指定
                poolsize=(2, 2))

    # 隠れ層への入力
    # 画像のピクセルをフラット化する
    # layer1の出力のサイズは (batch_size, 50, 4, 4) なのでflatten()によって
    # (batch_size, 50*4*4) = (batch_size, 800) になる
    layer2_input = layer1.output.flatten(2)

    # 全結合された隠れ層
    # 入力が800ユニット、出力が500ユニット
    layer2 = HiddenLayer(rng,
        input=layer2_input,
        n_in=50 * 4 * 4,
        n_out=500,
        activation=T.tanh)

    # 最終的な数字分類を行うsoftmax層
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

if __name__ == '__main__':
    evaluate_lenet5()
