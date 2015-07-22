#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from sklearn.datasets import fetch_mldata
import time
import cPickle
import pylab
import matplotlib.pyplot as plt

print "fetch MNIST dataset"
mnist = fetch_mldata("MNIST original")
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255

# 訓練データ数
N_train = 60000

x_train, x_test = np.split(mnist.data, [N_train])

# テストデータ数
N_test = x_test.shape[0]

# 学習済みのAutoencoderをロード
model = cPickle.load(open("autoencoder.pkl", "rb"))

def forward(x_data, train=False):
    x = chainer.Variable(x_data)
    y = F.sigmoid(model.l1(x))
    x_hat = F.sigmoid(model.l2(y))
    # 再構築した画像のデータを返す
    return x_hat.data

# テスト画像を可視化
plt.figure()
perm = np.random.permutation(N_test)
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.axis('off')
    plt.imshow(x_test[perm[i]].reshape((28, 28)), cmap=pylab.cm.gray_r)

# Autoencoderで100枚分の再構築した画像を得る
x_hat = forward(x_test[perm[0:100]])

# 再構築した画像を可視化
plt.figure()
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.axis('off')
    plt.imshow(x_hat[i].reshape((28, 28)), cmap=pylab.cm.gray_r)

# 最初の100個の重みを描画
n_hidden = model.l1.W.shape[0]
print n_hidden
plt.figure()
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.axis('off')
    plt.imshow(model.l1.W[i].reshape((28, 28)), cmap=pylab.cm.gray_r)
plt.show()
