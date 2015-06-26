#coding: utf-8
import cPickle
import pylab
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from logistic_sgd import load_data
from convolutional_mlp import LeNetConvPoolLayer

def visualize_filter(layer):
    """引数に指定されたLeNetConvPoolLayerのフィルタを描画する"""
    W = layer.W.get_value()
    n_filters, n_channels, h, w = W.shape
    plt.figure()
    pos = 1
    for f in range(n_filters):
        for c in range(n_channels):
            plt.subplot(n_filters, n_channels, pos)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.imshow(W[f, c], cmap=pylab.cm.gray_r)
            plt.axis('off')
            pos += 1
    plt.show()

def feedforward(input, layer, image_shape):
    """inputをlayerに通した畳み込み層の結果（畳み込み層の結果を可視化するため）と
    最終的な出力の特徴マップ（次の層への入力のため）を返す"""
    conv_out = conv.conv2d(input,
                           filters=layer.W,
                           filter_shape=layer.W.get_value().shape,
                           image_shape=image_shape)
    pooled_out = downsample.max_pool_2d(input=conv_out,
                                        ds=(2, 2),
                                        ignore_border=True)
    output = T.tanh(pooled_out + layer.b.dimshuffle('x', 0, 'x', 'x'))
    return conv_out, output

if __name__ == "__main__":
    layer0 = cPickle.load(open("layer0.pkl", "rb"))
    layer1 = cPickle.load(open("layer1.pkl", "rb"))
    print layer0, layer1

    # 最初の畳み込み層のフィルタの可視化
    visualize_filter(layer0)

    # 各層の出力の可視化
    input = T.tensor4()
    # 入力画像は1のためimage_shapeのbatch_size=1となる
    layer0_conv_out, layer0_out = feedforward(input, layer0, image_shape=(1, 1, 28, 28))
    layer1_conv_out, layer1_out = feedforward(layer0_out, layer1, image_shape=(1, 20, 12, 12))

    # 画像を1枚受けて、畳み込み層の出力を返す関数を定義
    f0 = theano.function([input], layer0_conv_out)
    f1 = theano.function([input], layer1_conv_out)

    # 入力はテストデータから適当な画像を一枚入れる
    datasets = load_data("mnist.pkl.gz")
    test_set_x, test_set_y = datasets[2]
    input_image = test_set_x.get_value()[0]
    layer0_conv_image = f0(input_image.reshape((1, 1, 28, 28)))
    layer1_conv_image = f1(input_image.reshape((1, 1, 28, 28)))

    # 畳み込み層の出力画像を可視化
    print layer0_conv_image.shape
    plt.figure()
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.imshow(layer0_conv_image[0, i], cmap=pylab.cm.gray_r)
    plt.show()

    print layer1_conv_image.shape
    plt.figure()
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.axis('off')
        plt.imshow(layer1_conv_image[0, i], cmap=pylab.cm.gray_r)
    plt.show()
