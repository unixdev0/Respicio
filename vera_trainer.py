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
import os
from vera_config import Config
from vera_config import Setup
from vera_img import ProcessImage
from vera_raw import ProcessRaw
import sys
from vera_nn import Classifier
from vera_common import Commons

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

class Config_NN:
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


def random_Epsilon_calculation(L_in, L_out, debug=False):
    epsilon = np.sqrt(6)/ np.sqrt(L_in + L_out)
    if debug:
        assert isinstance(epsilon, object)
        print("epsilon %f" % epsilon)
    return epsilon

def initialize_Weights(L_in, L_out, debug=False):
    epsilon_init = random_Epsilon_calculation(L_in, L_out, debug)
    Weights = 2 * epsilon_init * np.random.randn(L_in, L_out) - epsilon_init
    if debug:
        print("Weights %d x %d" % (L_in, L_out))
    return Weights

def build_model(conf=Config_NN):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = initialize_Weights(conf.nn_input_dim, conf.nn_hidden_dim)
    b1 = np.ones((conf.nn_hidden_dim, 1))
    W2 = initialize_Weights(conf.nn_hidden_dim, conf.nn_output_dim)
    b2 = np.ones((conf.nn_output_dim, 1))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model


def cost_J(model, h, y, m, conf=Config_NN):
    part1 = -y * np.log(h)
    part0 = (1 - y) * np.log(1 - h)
    diff = np.subtract(part1, part0)
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

def train_model(model, X, y, conf=Config_NN):
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

def gen_vocab():
    vera = Commons.getEnv(Config.vera)
    train_path = path.join(vera, Setup.path_train)
    for img_file in glob(path.join(train_path, '*.tiff')):
        head, tail = path.split(img_file)
        if tail.find('.tiff') != -1:
            tail = tail.replace('.tiff', '-raw.ai')
        raw_file = join(head, tail) # @FIXME: needs to work with multiple extensions
        processor = ProcessImage()
        processor.process_img(img_file, raw_file)
        head, tail = path.split(raw_file)
        if tail.find('-raw.ai') != -1:
            tail = tail.replace('-raw.ai', '-vera.ai')
        vera_file = join(head, tail)
        processor = ProcessRaw()
        processor.process_raw(raw_file, vera_file)
    #After all the images have been processed - generate the vocabulary
    vocab_processor = ProcessRaw()
    vocab_processor.process_vocab(train_path)

def run_training(labs):
    vera = Commons.getEnv(Config.vera)
    train_path = path.join(vera, Setup.path_train)
    classifier = Classifier()
    classifier.setup_traing(train_path)
    input_X = list()
    input_y = list()
    for vera_file in glob(path.join(train_path, '*-vera.ai')):
        x = classifier.load_data(vera_file)
        labels = np.zeros(len(labs), dtype=int)
        for label in labs:
            if label in vera_file:
                labels[labs.index(label)] = 1
                break
        input_X.append(x)
        input_y.append(labels)

    conf = Config_NN()
    conf.debug = True
    in_sz = len(input_X[0])
    out_sz = len(labels)
    conf.num_passes = 10000
    conf.num_samples = 10000
    conf.num_train = 8000
    conf.num_test = 2000
    conf.noise = 0.05
    conf.nn_input_dim = in_sz  # input layer dimensionality
    conf.nn_output_dim = out_sz  # output layer dimensionality
    conf.nn_hidden_dim = in_sz * 2
    conf.reg = 0.00005
    conf.alpha = 0.3
    X = np.array(input_X)
    y = np.array(input_y)
    model = build_model(conf)
    model, lowest_J = train_model(model, X, y, conf)
    model_path = path.join(vera, Setup.path_train, Setup.name_model)
    pickle.dump(model, open(model_path, 'wb'))


def main():
    args = sys.argv[1:]
#    gen_vocab()
    run_training(args)

if __name__ == "__main__":
    main()

'''
IMage Handling ===================
head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                while True:
                    try:
                        os.remove(new_path)
                        break
                    except Exception as e:
                        log.error('AI::VERA - exception %s', e.message)
                        time.sleep(1)
                        continue
                    except:
                        log.error('AI::VERA - unknown exception caught')
                        time.sleep(1)
                        continue
            os.rename(event.src_path, new_path)
            head, tail = path.split(new_path)
            if tail.find('.jpg') != -1:
                tail = tail.replace('.jpg', '-raw.ai')
            raw_file_path = join(self.nextPath, tail)
            processor = ProcessImage()
            processor.process_img(new_path, raw_file_path)

RawHandler ==========================================
head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                os.remove(new_path)
            os.rename(event.src_path, new_path)
            head, tail = path.split(new_path)
            if tail.find('-raw.ai') != -1:
                tail = tail.replace('-raw.ai', '-vera.ai')
            vera_file_path = join(self.nextPath, tail)
            processor = ProcessRaw()
            processor.process_raw(new_path, vera_file_path)

InputHandler =============================================
head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                os.remove(new_path)
            os.rename(event.src_path, new_path)
            pred = self.classifier.train(new_path)
'''