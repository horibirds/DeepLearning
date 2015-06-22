#coding: utf-8
import cPickle
import pylab
import matplotlib.pyplot as plt
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import load_data
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano
import theano.tensor as T

# フィルタの可視化

def visualize_learned_filters():
    fp = open("layer0.pkl", "r")
    layer0 = cPickle.load(fp)
    fp.close()

    fp = open("layer1.pkl", "r")
    layer1 = cPickle.load(fp)
    fp.close()

    print layer0
    print layer1

    plt.figure()
    for index in range(20):
        filter = layer0.W.get_value()[index, 0]
        plt.subplot(2, 10, index + 1)
        plt.axis('off')
        plt.imshow(filter, cmap=pylab.cm.gray_r)
    plt.show()

    plt.figure()
    for index in range(20):
        filter = layer1.W.get_value()[index, 0]
        plt.subplot(2, 10, index + 1)
        plt.axis('off')
        plt.imshow(filter, cmap=pylab.cm.gray_r)
    plt.show()


if __name__ == "__main__":
#    visualize_learned_filters()

    datasets = load_data("mnist.pkl.gz")
    test_set_x, test_set_y = datasets[2]

    input = T.tensor4()
    input_image = test_set_x.get_value()[0]
#     plt.figure()
#     plt.imshow(input.reshape((28, 28)), cmap=pylab.cm.gray_r)
#     plt.show()

    fp = open("layer0.pkl", "r")
    layer0 = cPickle.load(fp)
    fp.close()

    conv_out = conv.conv2d(input,
                           filters=layer0.W,
                           filter_shape=layer0.W.get_value().shape,
                           image_shape=(1, 1, 28, 28))
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=(2, 2),
        ignore_border=True)
    layer0_output = T.tanh(pooled_out + layer0.b.dimshuffle('x', 0, 'x', 'x'))
    f = theano.function([input], layer0_output)
    layer0_image = f(input_image.reshape((1, 1, 28, 28)))
    plt.figure()
    for index in range(20):
        plt.subplot(2, 10, index + 1)
        plt.axis('off')
        plt.imshow(layer0_image[0, index], cmap=pylab.cm.gray_r)
    plt.show()

    fp = open("layer1.pkl", "r")
    layer1 = cPickle.load(fp)
    fp.close()




    conv_out = conv.conv2d(input,
                           filters=layer1.W,
                           filter_shape=layer1.W.get_value().shape,
                           image_shape=(1, 20, 12, 12))
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=(2, 2),
        ignore_border=True)
    layer1_output = T.tanh(pooled_out + layer1.b.dimshuffle('x', 0, 'x', 'x'))
    f = theano.function([input], layer1_output)
    layer1_image = f(layer0_image.reshape((1, 20, 12, 12)))

    plt.figure()
    for index in range(50):
        print layer1_image.shape
        plt.subplot(5, 10, index + 1)
        plt.axis('off')
        plt.imshow(layer1_image[0, index], cmap=pylab.cm.gray_r)
    plt.show()
