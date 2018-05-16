#-*- coding:utf-8 -*-

import numpy as np
import sys

sys.path.append("../")

class LinearClassifier(object):
    """线性分类器
    """
    def __init__(self):
        self.W = None

    def train(self, X, y, lr=0.001, reg=1e-5, num_iters=100,
        batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W += -lr * grad
            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass

def softmax_loss_loop(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_trains = X.shape[0]
    num_classes = W.shape[1]
    for i in xrange(num_trains):
        scores = X[i].dot(W)
        max_score = np.max(scores)
        scores -= max_score
        scores = np.exp(scores)
        normal = np.sum(scores)
        scores /= normal
        loss += -1 * np.log(scores[y[i]])
        for k in range(num_classes):
            p_k = scores[k]
            dW[:, k] += ((p_k - (k == y[i])) * X[i]).T
    loss /= num_trains
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_trains
    dW += reg*W
    return loss, dW

def softmax_loss_noloop(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f
    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    return loss, dW

class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_loop(self.W, X_batch, y_batch, reg)
        #return softmax_loss_noloop(self.W, X_batch, y_batch, reg)
