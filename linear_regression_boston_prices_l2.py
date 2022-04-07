#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import sklearn.datasets as datasets
import numpy as np

def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    y_hat = np.dot(X, w)
    Nsample = len(y)
    loss = (1/(2*Nsample))*np.sum(np.square(y - y_hat))
    risk = (1/Nsample)*np.sum(np.absolute(y - y_hat))

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val, l):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = w
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):
        i = 0
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):
            i += 1
            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            M = len(y_batch)
            g = (1/M)*((X_batch.T).dot(y_hat_batch - y_batch)) + l*w
            w = w - alpha*g


    
        avg_loss = loss_this_epoch/i

        losses_train.append(avg_loss)
        _, loss, risk = predict(X_val, w, y_val)
        if risk < risk_best:
            w_best = w
            risk_best = risk
            loss_best = loss
            epoch_best = epoch
        risks_val.append(risk)

    print("Best Epoch:", epoch_best)
    print("Best Risk Validation:", risk_best)
    print("Best Loss Validation:", loss_best)

    # Return some variables as needed
    return w_best, losses_train, risks_val, risk_best

def holdout(X_train, y_train, X_val, y_val, hyperparams):
    risk_l_best = 100000
    l_best = None
    for l in hyperparams:
        print("Current Lamda: ", l)
        X_train_l = X_train[:200]
        y_train_l = y_train[:200]
        X_val_l = X_train[200:]
        y_val_l = y_train[200:]
        w_best, _, _, _ = train(X_train_l, y_train_l, X_val_l, y_val_l, l)
        _, _, risk_l = predict(X_val, w_best,y_val)
        print(risk_l, risk_l_best)
        if risk_l < risk_l_best:
            risk_l_best = risk_l
            l_best = l
        print("============================")
    return l_best



############################
# Main code starts here
############################
# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
X_ = np.concatenate((X_, np.square(X_)), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting
k = 10
alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay
hypers = [3, 1, 0.3, 0.1, 0.03, 0.01]

# TODO: Your code here
l = holdout(X_train, y_train, X_val, y_val, hypers)
print("Best l:",l)
w_best, losses_train, risks_val, _ = train(X_train, y_train, X_val, y_val, l)

# Perform test by the weights yielding the best validation performance
y_hat, loss, risk = predict(X_test, w_best, y_test)
print("Testing Loss:", loss, "Testing Risk:", risk)
x = list(range(0,100))
# Report numbers and draw plots as required.
plt.scatter(x, losses_train)
plt.title('Learning Curve of Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_learning_curve_non_linear_l2')

plt.figure()
plt.scatter(x, risks_val)
plt.title('Learning Curve of Validation Risk')
plt.xlabel('Epoch')
plt.ylabel('Average Risk')
plt.savefig('risk_learning_curve_non_linear_l2')

