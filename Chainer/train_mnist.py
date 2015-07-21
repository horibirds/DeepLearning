#coding: utf-8
import numpy as np
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from sklearn.datasets import fetch_mldata

"""
Chainer example
train a multi-layer perceptron on MNIST
https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py
"""

batchsize = 100
n_epoch = 20
n_units = 1000

print "load MNIST dataset"
mnist = fetch_mldata('MNIST original')
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'], [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, 10))

# neural network architecture
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

fp1 = open("train_loss.txt", "w")
fp2 = open("test_loss.txt", "w")

# learning loop
for epoch in xrange(1, n_epoch + 1):
    print "epoch", epoch

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    fp1.write("%f\t%f\n" % (sum_loss / N, sum_accuracy / N))
    fp1.flush()

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    fp2.write("%f\t%f\n" % (sum_loss / N_test, sum_accuracy / N_test))
    fp2.flush()

fp1.close()
fp2.close()
