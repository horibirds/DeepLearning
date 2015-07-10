#coding: utf-8
import os
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data

try:
    import PIL.Image as Image
except ImportError:
    import Image

class AutoEncoder(object):
    """自己符号化器"""
    def __init__(self, numpy_rng, theano_rng=None,
                 input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            # 入力層と出力層の間の重み
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6.0 / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            # 入力層（visible）のユニットのバイアス
            bvis = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                borrow=True)

        if not bhid:
            # 出力層（hidden）のユニットのバイアス
            bhid = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='b',
                borrow=True)

        # 順方向での重み
        self.W = W

        # bはhiddenのバイアス
        self.b = bhid

        # b'はvisibleのバイアス
        self.b_prime = bvis

        # 逆方向での重み
        # tied weightsの制約あり
        self.W_prime = self.W.T

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """入力層の値を隠れ層の値に変換"""
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """隠れ層の値を入力層の値に逆変換"""
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        """コスト関数と更新式のシンボルを返す"""
        # 入力を変換
        y = self.get_hidden_values(self.x)

        # 変換した値を逆変換で入力に戻す
        z = self.get_reconstructed_input(y)

        # コスト関数のシンボル
        # 元の入力と再構築した入力の交差エントロピーの平均を誤差関数とする
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # 誤差関数の微分
        gparams = T.grad(cost, self.params)

        # 更新式のシンボル
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return cost, updates

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """img_shapeのサイズの画像をtile_shapeの形に並べる
    tile_spacingは画像と画像の間のスペースのサイズ"""
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp
                 for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

def test_dA():
    learning_rate = 0.1
    training_epochs = 15
    batch_size = 20
    output_dir = 'dA_plots'

    # 学習データのロード
    # 今回は評価はしないため訓練データのみ
    # TODO:テストデータの圧縮・復元がどれくらいうまくいくのかテストしてみる
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]

    # ミニバッチ数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # ミニバッチのインデックスを表すシンボル
    index = T.lscalar()

    # ミニバッチの学習データを表すシンボル
    x = T.matrix('x')

    # 出力先がなかったら作って移動
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    # モデル構築
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    auto_encoder = AutoEncoder(numpy_rng=rng,
                               theano_rng=theano_rng,
                               input=x,
                               n_visible=28 * 28,
                               n_hidden=500)

    # コスト関数と更新式のシンボルを取得
    cost, updates = auto_encoder.get_cost_updates(learning_rate=learning_rate)

    # 訓練用の関数を定義
    train_da = theano.function([index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        })

    # モデル訓練
    start_time = time.clock()
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print "Training epoch %d, cost %f" % (epoch, np.mean(c))

    end_time = time.clock()
    training_time = (end_time - start_time)

    print "The no corruption code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((training_time / 60.0))

    # 学習された重みを可視化
    tile_images = tile_raster_images(X=auto_encoder.W.get_value(borrow=True).T,
                                     img_shape=(28, 28),
                                     tile_shape=(10, 10),
                                     tile_spacing=(1, 1))

if __name__ == "__main__":
    test_dA()
