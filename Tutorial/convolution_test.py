#coding: utf-8
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import pylab
from PIL import Image

rng = np.random.RandomState(23455)

input = T.tensor4(name='input')

# フィルタの重み行列を作成
# 3枚のfeature mapを2枚のfeature mapに変換する
# フィルタのサイズは9x9
# 今回はテストなので学習せずに単に乱数で初期化
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3 * 9 * 9)
W = theano.shared(np.asarray(
    rng.uniform(low=-1.0 / w_bound, high=1.0 / w_bound, size=w_shp),
    dtype=input.dtype), name='W')

# バイアスベクトルを作成
# 今回はテストなので学習せずに単に乱数で初期化
# 行先が2枚のfeature mapなのでサイズは2
b_shp = (2, )
b = theano.shared(np.asarray(
    rng.uniform(low=-0.5, high=0.5, size=b_shp),
    dtype=input.dtype), name='b')

conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)


img = Image.open("3wolfmoon.jpg")

# (height, width, channel)
img = np.asarray(img, dtype='float64') / 256.0
print img.shape

# (channel, height, width)
img_ = img.transpose(2, 0, 1)
print img_.shape

# 4Dテンソルに変換 (dummy, channel, height, width)
img_ = img_.reshape(1, 3, 639, 516)
print img_.shape

# フィルタをかける
filtered_img = f(img_)

# オリジナル画像と各feature mapの画像を表示
pylab.subplot(1, 3, 1)
pylab.axis('off')
pylab.imshow(img)

pylab.subplot(1, 3, 2)
pylab.axis('off')
pylab.imshow(filtered_img[0, 0, :, :])

pylab.subplot(1, 3, 3)
pylab.axis('off')
pylab.imshow(filtered_img[0, 1, :, :])

pylab.show()

