from __future__ import division
import sys
from sys import path
import os
import os.path as path
import nltk, re, pprint
from nltk.corpus import wordnet
from os.path import join
from glob import glob
try:
    import Image
except ImportError:
    from PIL import Image
    import pytesseract
import nltk.data

def process_txt(txt):
    flag = False
    try:
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        s = ''
        if len(txt) > 0:
            txt = txt.lower()
            txt = re.sub('[^a-zA-Z\d\s:]', ' ', txt)
            txt = re.sub('[\s:]', ' ', txt)
            txt = re.sub('\s{2,}', ' ', txt)

            if txt.find(' ') == 0:
                txt = txt[1:]
            if txt.rfind(' ') == len(txt) - 1:
                txt = txt[:-1]
            words = nltk.word_tokenize(txt)
            for word in words:
                if len(word) <= 3:
                    continue
                if word.isdigit():
                    continue
                try:
                    isWord = wordnet.synsets(word)
                    if len(isWord) == 0:
                        continue
                except:
                    print 'Exception'
                s += word
                s += ' '
            if s.rfind(' ') == len(s) - 1:
                s = s[:-1]

        flag = True
    except:
        flag = False
    return flag, s

def process_vocab():
    dir = 'C:/Workspace/Bills/input'
    ext = '*-input.txt'
    d = set([])
    dirs = ['train', 'test']
    for d1 in dirs:
        vocab = set([])
        files = glob(join(dir, d1, ext))
        for f in files:
            d = set([])
            if path.isfile(f) == True:
                txtfile = open(f).readlines()
                raw = list()
                for line in txtfile:
                    raw.append(line)
                if len(raw) > 0:
                    d = d | set(raw)
            vocab = vocab | d
        head, tail = path.split(f)
        txtfilepath = join(head, 'Respicio-pp.txt')
        txtf = open(txtfilepath, 'w+')
        for item in vocab:
            txtf.write(item)
        txtf.close()

def process_inputs():
    dir = 'C:/Workspace/Bills/input'
    ext = '*.txt'
    inputs = list()
    dirs = ['train', 'test']
    for d1 in dirs:
        files = glob(join(dir, d1, ext))
        for f in files:
            d = set([])
            if path.isfile(f) == True:
                txtfile = open(f).readlines()
                raw = list()
                for line in txtfile:
                    flag, txt = process_txt(line)
                    if flag:
                        if len(txt) > 2:
                            raw.append(txt)
                if len(raw) > 0:
                    for sentence in raw:
                        words = nltk.word_tokenize(sentence)
                        d = d | set(words)
            sd = set(sorted(d))
            vocab = set([])
            porter = nltk.PorterStemmer()
            for stemming in sd:
                stemmed_word = porter.stem(stemming)
                vocab.add(stemmed_word)
            head, tail = path.split(f)
            find_idx = tail.rfind('.txt')
            if find_idx != -1:
                s = tail.replace('.txt', '-input.txt')
            txtfilepath = path.join(head, s)
            txtf = open(txtfilepath, 'w+')
            for item in vocab:
                txtf.write(item)
                txtf.write('\n')
            txtf.close()

def clean_up():
    dir = 'C:/Workspace/Bills/input'
    ext = '*[input|pp].txt'
    inputs = list()
    d = set([])
    dirs = ['train', 'test']
    for d1 in dirs:
        files = glob(join(dir, d1, ext))
        for f in files:
            os.remove(f)

def main():
    clean_up()
    process_inputs()
    process_vocab()


if __name__ == "__main__":
    main()