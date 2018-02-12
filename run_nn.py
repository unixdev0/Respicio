import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import os.path as path
import re, pprint
import itertools
import operator
from glob import glob
from os.path import join
import pickle

def nonlin(x, deriv=False):
    if Config.activation == 'sigmoid':
        if(deriv==True):
            return x*(1-x)
        else:
            return 1/(1+np.exp(-x))
    elif Config.activation == 'tanh':
        if(deriv==True):
            return 1.0 - x**2
        else:
            return np.tanh(x)

class Config:
    nn_input_dim = 5  # input layer dimensionality
    nn_output_dim = 4  # output layer dimensionality
    nn_hidden_dim = 5
    num_passes = 30000
    num_samples=10000
    num_train=9500
    num_test=500
    noise=0.05
    debug = True
    reg = 0.0001
    alpha = 0.9
    activation = 'sigmoid'
    epsilon = 1e-4
    epsilon_init = 0.12
    moons = True

def generate_data(conf=Config):
    np.random.seed(0)
    if conf.moons == True:
        X, y = datasets.make_moons(conf.num_samples, noise=0.20)
    else:
        #X, y = make_multilabel_classification(n_classes=4, n_samples=conf.num_samples, n_features=conf.nn_input_dim, n_labels=1,
        #                                      allow_unlabeled=False,
        #                                      random_state=1)
        X, y = datasets.make_classification(n_samples=conf.num_samples, n_features=conf.nn_input_dim, n_informative=2, n_redundant=2, n_repeated=0,
                                             n_classes=conf.nn_output_dim, n_clusters_per_class=1, weights=None, flip_y=0.01,
                                             class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True,
                                             random_state=1)
    return X, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def random_Epsilon_calculation(L_in, L_out, debug=False):
    epsilon = np.sqrt(6)/ np.sqrt(L_in + L_out)
    if debug:
        assert isinstance(epsilon, object)
        print("epsilon %f" % epsilon)
    return epsilon

def initialize_Weights(L_in, L_out, debug=False):
    epsilon_init = random_Epsilon_calculation(L_in, L_out, debug)
    Weights = 2 * epsilon_init * np.random.randn(L_in, L_out) - epsilon_init
    #Weights = np.random.randn(L_in, L_out) / np.sqrt(L_in)
    if debug:
        print("Weights %d x %d" % (L_in, L_out))
    return Weights

def build_model(conf=Config):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = initialize_Weights(conf.nn_input_dim, conf.nn_hidden_dim)
    b1 = np.ones((conf.nn_hidden_dim, 1))
    W2 = initialize_Weights(conf.nn_hidden_dim, conf.nn_output_dim)
    b2 = np.ones((conf.nn_output_dim, 1))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model


def cost_J(model, h, y, m, conf=Config):
    part1 = -y * np.log(h)
    part0 = (1 - y) * np.log(1 - h)
    diff = np.subtract(part1, part0)
    #diff = np.sum(-y * np.log(h))/m
    J_unreg = np.sum(diff)/m
    W1 = model['W1']
    W2 = model['W2']
    p1 = np.sum( np.power(W1, 2) )
    p2 = np.sum( np.power(W2, 2) )
    reg = conf.reg * ( p1 + p2 ) / (2 * m)
    J = J_unreg + reg
    if conf.debug == True:
        print "Cost J = ", J, ", reg = ", reg
    return J

def train_model(model, X, y, conf=Config):
    # Initialize the parameters to random values. We need to learn these.
    m = len(X)
    lowest_J = tuple((999,0))
    # Gradient descent. For each batch...
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    for i in range(0, conf.num_passes):
        # Forward propagation
        a1 = X
        z2 = np.dot(a1, W1) + b1.T
        a2 = nonlin(z2)
        z3 = np.dot(a2, W2) + b2.T
        a3 = nonlin(z3)
        #z4 = np.dot(a3, W3) + b3.T
        #a4 = nonlin(z4)

        d3 = a3 - y
        d2 = d3.dot(W2.T) * nonlin(a2, True)

        grad2 = (d3.T.dot(a2) + conf.reg * (np.sum(W2))) / m
        grad1 = (d2.T.dot(a1) + conf.reg * (np.sum(W1))) / m

        W1 -= conf.alpha * grad1.T
        W2 -= conf.alpha * grad2.T

        db2 = np.sum(d3, axis=0) / m
        db1 = np.sum(d2, axis=0) / m

        b1 -= conf.alpha * db1.reshape(conf.nn_hidden_dim, 1)
        b2 -= conf.alpha * db2.reshape(conf.nn_output_dim, 1)

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if i % 1000 == 0:
            J = cost_J(model, a3, y, m, conf)
            if conf.debug:
                print("Loss after iteration %i: %f" % (i, J))
            if J < lowest_J[0]:
                lowest_J = (J, i)
    return model, lowest_J


def predict(model, x, conf=Config):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    a1 = x
    z2 = np.dot(a1, W1) + b1.T
    a2 = nonlin(z2)
    z3 = np.dot(a2, W2) + b2.T
    a3 = nonlin(z3)
    #a3.reshape(1,conf.nn_output_dim)
    pred = a3
    return pred

def gen_train_data(conf=Config):
    dir = 'C:/Workspace/Bills/input/train'
    vocab_name = 'Respicio-vocab.txt'
    ext = '*-input.txt'
    inputs_X = list()
    inputs_y = list()
    #load vocabulary set
    vocab = list()
    vocab_file_name = join(dir, vocab_name)
    if path.isfile(vocab_file_name):
        temp = open(vocab_file_name).read()
        vocab = re.split('\s', temp)
    files = glob(join(dir, ext))
    for f in files:
        label = np.zeros(len(files), dtype=int)
        features = np.zeros(len(vocab), dtype=int)
        if path.isfile(f) == True:
            words = re.split('\s', open(f).read())
            for word in words:
                try:
                    idx = vocab.index(word)
                    if idx != -1:
                        features[idx] = 1
                except:
                    print "ERROR: Word [", word, "] not in the vocabulary"
        label[files.index(f)] = 1
        inputs_X.append(features)
        inputs_y.append(label)
    if conf.debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(inputs_X)
        pp.pprint(inputs_y)
    return inputs_X, inputs_y, vocab

def gen_test_data(vocab, conf=Config):
    dir = 'C:/Workspace/Bills/input/test'
    ext = '*-input.txt'
    inputs_X = list()
    #load vocabulary set
    files = glob(join(dir, ext))
    for f in files:
        label = np.zeros(len(files), dtype=int)
        features = np.zeros(len(vocab), dtype=int)
        if path.isfile(f) == True:
            words = re.split('\s', open(f).read())
            for word in words:
                try:
                    idx = vocab.index(word)
                    if idx != -1:
                        features[idx] = 1
                except:
                    print "ERROR: Word [", word, "] not in the vocabulary"
        inputs_X.append(features)
    if conf.debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(inputs_X)
    return inputs_X

def gen_test(conf=Config):
    dir = 'C:/Workspace/Bills'
    files = [
        'Doc-Respicio-Test-001',
        'Doc-Respicio-Test-002',
        'Doc-Respicio-Test-003'
    ]
    vocab_name = 'Respicio-pp'
    ext = 'txt'
    inputs_X = list()
    inputs_y = list()
    #load vocabulary set
    vocab = list()
    vocab_file_name = dir + '/' + vocab_name + '.' + ext
    if path.isfile(vocab_file_name):
        temp = open(vocab_file_name).read()
        vocab = re.split('\s', temp)

    for f in files:
        label = np.zeros(len(files), dtype=int)
        features = np.zeros(len(vocab), dtype=int)
        filepath = dir + '/' + f + '.' + ext
        if path.isfile(filepath) == True:
            words = re.split('\s', open(filepath).read())
            for word in words:
                idx = vocab.index(word)
                if idx != -1:
                    features[idx] = 1
        inputs_X.append(features)
    if conf.debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(inputs_X)
    return inputs_X

classes = ('Enbridge Gas','Toronto Utility','Toronto Hydro','T-Mobile')

def main():
    conf = Config()
    conf.debug = True

    inputs_X, inputs_y, vocab = gen_train_data(conf)
    in_sz = len(inputs_X[0])
    out_sz = len(inputs_y[0])
    conf.num_passes = 10000
    conf.num_samples = 10000
    conf.num_train = 8000
    conf.num_test = 2000
    conf.reg = 0.
    conf.noise = 0.05
    conf.moons = False
    conf.nn_input_dim = in_sz  # input layer dimensionality
    conf.nn_output_dim = out_sz  # output layer dimensionality
    conf.nn_hidden_dim = in_sz * 2

    J_history = list()
    alpha = [0.18, 0.20, 0.25, 0.3, 0.35]

    X = np.array(inputs_X)
    y = np.array(inputs_y).T

    #alpha = [0.12]
    reg = [0.]
    #reg = [ 0, 0.00005, 0.00005, 0.0005, 0.005, 0.05, 0.5]
    for r in reg:
        model = build_model(conf)
        conf.alpha = 0.3
        conf.reg = r
        #X, y = generate_data(conf)

        if conf.moons == True:
            y_v = np.eye(conf.nn_output_dim)[y, :]
        else:
            #y_v = y.copy()
            #y_v = np.eye(conf.nn_output_dim, dtype=int)[y, :]
            y_v = y.copy()
        #X_train = X[0:conf.num_train, :].copy()
        #y_v_train = y_v[0:conf.num_train, :].copy()
        #X_pred = X[conf.num_train:conf.num_train + conf.num_test, :].copy()
        #y_pred = y_v[conf.num_train:conf.num_train + conf.num_test, :].copy()
        model, lowest_J = train_model(model, X, y_v, conf)
        error = 0.
        X_test = gen_test_data(vocab, conf)
        pred = predict(model, X_test, conf)
        pred_prob = pred.copy()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # print "Prediction idx:[", pred, "] in ", y_v[i, :], " = ",  y_v[i, pred], " - factual: ", y[i]
        for i in range(len(pred)):
            print "Prediction=", classes[np.argmax(pred[i, :])], " with confidence=", np.max(pred_prob[i, :])
        pickle.dump(model, open('model.vera', 'wb'))

        model = pickle.load(open('model.vera', 'rb'))


if __name__ == "__main__":
    main()