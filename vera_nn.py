import numpy as np
import os
from os import environ
import os.path as path
import re, pprint
import itertools
import operator
from glob import glob
from os.path import join
import time
import logging as log
import sys
import pickle
from vera_config import Config
from vera_config import Setup
from vera_common import Commons
import json

class Classifier(object):

    #classes = ('Enbridge Gas', 'Toronto Utility', 'Toronto Hydro', 'T-Mobile')
    #classes = ('Fido', 'TekSavvy', 'Google')

    def __init__(self):
        log.info('AI::VERA - Instance %s has been created', type(self).__name__)
        vera = Commons.getEnv(Config.vera)
        model_path = path.join(vera, Setup.path_model, Setup.model_config)
        log.info('AI::VERA - Loading model metadata for type <<< %s >>>', Setup.model.upper())
        if not path.exists(model_path):
            #@TODO: handle differently to indicate that initialization failed
            log.error('AI::VERA - Cannot find model metadata file %s', model_path)
            return
        meta = json.load(open(model_path, 'r'))
        for m in meta['models']:
            if Setup.model in m['name']:
                self.classes = tuple(m['labels'])
                self.vocab = m['vocab']
                self.model = m['model']
                break

    def nonlin(self, x, deriv=False):
        log.info('AI::VERA - %s.nonlin activation=%s, derivative=%r', \
                 type(self).__name__, Config.activation, deriv)
        if Config.activation == 'sigmoid':
            if deriv:
                return x*(1-x)
            else:
                return 1/(1+np.exp(-x))
        elif Config.activation == 'tanh':
            if deriv:
                return 1.0 - x**2
            else:
                return np.tanh(x)

    def classify(self, x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        a1 = x
        z2 = np.dot(a1, W1) + b1.T
        a2 = self.nonlin(z2)
        z3 = np.dot(a2, W2) + b2.T
        a3 = self.nonlin(z3)
        return a3

    def load_data(self, path_data):
        X = list()
        features = np.zeros(len(self.vocab), dtype=int)
        if path.isfile(path_data):
            words = re.split('\s', open(path_data).read())
            for word in words:
                try:
                    idx = self.vocab.index(word)
                except ValueError:
                    continue
                if idx != -1:
                    features[idx] = 1
        X.extend(features)
        return X

    def load_vocab(self, vera):
        vocab = list()
        vocab_path = path.join(vera, Setup.path_model, self.vocab)
        if path.isfile(vocab_path):
            temp = open(vocab_path).read()
            vocab = re.split('\s', temp)
        return vocab

    def load_vocab_from_path(self, vocab_path):
        vocab = list()
        if path.isfile(vocab_path):
            temp = open(vocab_path).read()
            vocab = re.split('\s', temp)
        return vocab

    def load_model(self, vera):
        model = dict()
        path_model = path.join(vera, Setup.path_model, self.model)
        if path.isfile(path_model):
            model = pickle.load(open(path_model, 'rb'))
        return model

    def setup_traing(self, path_to_traing_data):
        self.vera = Commons.getEnv(Config.vera)
        vocab_path = path.join(self.vera, Setup.path_train, Setup.name_vocab)
        self.vocab = self.load_vocab_from_path(vocab_path)

    def setup(self):
        vera = Commons.getEnv(Config.vera)
        self.model = self.load_model(vera)
        self.vocab = self.load_vocab(vera)

    def process(self, path_data):
        X = self.load_data(path_data)
        pred = self.classify(X)
        pred_prob = pred.copy()
        pred[pred >= Config.threshold] = 1
        pred[pred < Config.threshold] = 0
        predictions = list()
        for i in range(len(pred)):
            predictions.append((self.classes[np.argmax(pred[i, :])], np.max(pred_prob[i, :])))
        return predictions

