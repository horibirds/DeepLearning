#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from sklearn.datasets import fetch_mldata
import time
import cPickle

"""
Simple Autoencoder
https://gist.github.com/matsuken92/3b945f3ea4d07e9dcc0a
"""

USE_GPU = False

batchsize = 100
n_epoch = 30
n_units = 500
noised = False

# Autoencoderは教師なし学習なのでmnist.targetのラベルデータは使わない
print "fetch MNIST dataset"
mnist = fetch_mldata("MNIST original")
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255

# 訓練データ数
N_train = 60000

x_train, x_test = np.split(mnist.data, [N_train])

# テストデータ数
N_test = x_test.shape[0]

# Autoencoder model
model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                            l2=F.Linear(n_units, 784))

if USE_GPU:
    cuda.init()
    model.to_gpu()

def forward(x_data, train=True):
    # 入力画像
    x = chainer.Variable(x_data)

    # 隠れ層へ写像した入力画像
    y = F.sigmoid(model.l1(x))

    # 隠れ層から再構築した入力画像
    x_hat = F.sigmoid(model.l2(y))

    # 再構築した入力画像と元の入力画像の二乗誤差を最小化したい誤差関数とする
    return F.mean_squared_error(x_hat, x)

# Setup Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# 各エポックでの訓練データの誤差とテストデータの誤差を記録しておくリスト
train_loss = []
test_loss = []

fp1 = open("train_loss.txt", "w")
fp2 = open("test_loss.txt", "w")

# Learning loop
start_time = time.clock()

for epoch in xrange(1, n_epoch + 1):
    print "epoch", epoch

    # 訓練
    perm = np.random.permutation(N_train)
    sum_loss = 0
    for i in xrange(0, N_train, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        if USE_GPU:
            x_batch = cuda.to_gpu(x_batch)

        optimizer.zero_grads()
        loss = forward(x_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

    print "train mean loss = %f" % (sum_loss / N_train)
    fp1.write("%d\t%f\n" % (epoch, sum_loss / N_train))
    fp1.flush()

    # 評価
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]

        if USE_GPU:
            x_batch = cuda.to_gpu(x_batch)

        # 評価の時はtrainをFalseにしてパラメータ更新を抑制する
        loss = forward(x_batch, train=False)

        test_loss.append(loss.data)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize

    print "test mean loss = %f" % (sum_loss / N_test)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N_test))
    fp2.flush()

fp1.close()
fp2.close()

end_time = time.clock()
print "training time: %fm" % ((end_time - start_time) / 60.0)

# 学習したモデルをダンプ
cPickle.dump(model, open("autoencoder.pkl", "wb"), protocol=2)
