#coding: utf-8
import numpy as np
import cPickle
import gzip
from sklearn import linear_model, datasets
from sklearn.metrics.metrics import confusion_matrix, classification_report

# scikit-learnを用いたロジスティック回帰

def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == "__main__":
    datasets = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print train_set_x.shape
    print train_set_y.shape

    logreg = linear_model.LogisticRegression()
    logreg.fit(train_set_x, train_set_y)
    predictions = logreg.predict(test_set_x)
    print confusion_matrix(test_set_y, predictions)
    print classification_report(test_set_y, predictions)
