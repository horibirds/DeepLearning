#coding: utf-8
import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """隠れ層の初期化
        rng: 乱数生成器（重みの初期化で使用）
        input: ミニバッチ単位のデータ行列（n_samples, n_in)
        n_in: 入力データの次元数
        n_out: 隠れ層のユニット数
        W: 隠れ層の重み
        b: 隠れ層のバイアス
        activation: 活性化関数
        """
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(6.0 / (n_in + n_out)),
                            high=np.sqrt(6.0 / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=theano.config.floatX)


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # 多層パーセプトロンは隠れ層とロジスティック回帰で表される出力層から成る
        # ロジスティック回帰への入力は隠れ層の出力になる点に注意
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
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

    # 乱数生成器
    rng = np.random.RandomState(1234)

    # MLPを構築
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)

if __name__ == "__main__":
    test_mlp()

