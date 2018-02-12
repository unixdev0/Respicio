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

regex_email = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
regex_web   = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
regex_phone = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
regex_url   = re.compile(r"(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

def is_email(word):
    is_match = regex_email.match(word)
    return is_match

def is_phone_num(word):
    is_match = regex_phone.match(word)
    return is_match

def is_web_address(word):
    is_match = regex_web.match(word)
    return is_match

def is_url(word):
    is_match = regex_url.match(word)
    return is_match

def is_key_word(word):
    flag = False
    if is_email(word):
        flag = True
    elif is_url(word):
        flag = True
    elif is_phone_num(word):
        flag = True
    elif is_web_address(word):
        flag = True
    return flag

def get_keywords(txt):
    if is_email(txt):
        keywords = regex_email.findall(txt)
    elif is_url(txt):
        keywords = regex_url.findall(txt)
    elif is_phone_num(txt):
        keywords = regex_phone.findall(txt)
    elif is_web_address(txt):
        keywords = regex_web.findall(txt)
    else:
        keywords = []
    return keywords

def process_txt(txt):
    flag = False
    try:
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        s = ''
        if len(txt) > 0:
            txt = txt.lower()
            txt = re.sub('[^a-zA-Z\d\s\@\.\-():]', ' ', txt)
            txt = re.sub('[\s:]', ' ', txt)
            txt = re.sub('\s{2,}', ' ', txt)

            if txt.find(' ') == 0:
                txt = txt[1:]
            if txt.rfind(' ') == len(txt) - 1:
                txt = txt[:-1]
            emails = regex_email.findall(txt)
            if len(emails) > 0:
                for email in emails:
                    s += email
            words = nltk.word_tokenize(txt)
            for word in words:
                if len(word) <= 3:
                    continue
                if not is_key_word(word):
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
    ext = '*-raw.txt'
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
                    emails = regex_email.findall(line)
                    if len(emails):
                        for email in emails:
                            raw.append(email)
                    else:
                        flag, txt = process_txt(line)
                        if flag:
                            if len(txt) > 2:
                                raw.append(txt)
                if len(raw) > 0:
                    for sentence in raw:
                        emails = regex_email.findall(sentence)
                        if len(emails):
                            words = emails
                        else:
                            words = nltk.word_tokenize(sentence)
                        d = d | set(words)
            sd = set(sorted(d))
            vocab = set([])
            porter = nltk.PorterStemmer()
            for stemming in sd:
                if not is_key_word(stemming):
                    stemmed_word = porter.stem(stemming)
                else:
                    stemmed_word = stemming
                vocab.add(stemmed_word)
            head, tail = path.split(f)
            find_idx = tail.rfind('.txt')
            if find_idx != -1:
                s = tail.replace('-raw.txt', '-input.txt')
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