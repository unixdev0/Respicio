from __future__ import division
import sys
from sys import path
import os
import os.path as path
import nltk, re, pprint
from nltk.corpus import wordnet
import numpy as np
import pprint

def main():
    dir = 'C:/Workspace/Bills'
    files = [
        'Doc-Respicio-Test-001',
        'Doc-Respicio-Test-002',
        'Doc-Respicio-Test-003'
    ]
    vocab_name = 'Respicio-pp'
    ext = 'txt'
    inputs = list()
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
        label[files.index(f)] = 1
        inputs.append((features, label))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(inputs)

def gen_input():
    dir = 'C:/Workspace/Bills'
    files = [
        'Doc-Respicio-Test-001',
        'Doc-Respicio-Test-002',
        'Doc-Respicio-Test-003'
    ]
    vocab_name = 'Respicio-pp'
    ext = 'txt'
    inputs = list()
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
        label[files.index(f)] = 1
        inputs.append((features, label))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(inputs)

if __name__ == "__main__":
    main()