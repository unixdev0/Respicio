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
regex_web   = re.compile(r"(([a-z]{1,3}\.)?([a-z]+\.[a-z]{1,3}))")
regex_phone = re.compile(r"\s(((\+?1[-\.\s]?)?(\d{3}[-\.\s]?))(\d{3}[-\.\s]?\d{2}[-\.\s]?\d{2}))\s?")
#regex_url   = re.compile(r"(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

def is_email(word):
    is_match = regex_email.search(word)
    if not is_match is None:
        someFlag = True
    return not (is_match is None)

def is_phone_num(word):
    is_match = regex_phone.search(word)
    return not (is_match is None)

def is_web_address(word):
    is_match = regex_web.search(word)
    return not (is_match is None)

def is_key_word(word):
    flag = False
    if is_email(word):
        flag = True
    elif is_phone_num(word):
        flag = True
    elif is_web_address(word):
        flag = True
    return flag

def get_keywords(txt):
    keywords = []
    matches = []
    tmp = str(txt)
    the_rest = str(txt)

    if is_email(tmp):
        matches = regex_email.findall(tmp)
        if len(matches) > 0:
            emails = []
            emails.append(matches[0])
            for c, x in enumerate(emails):
                idx = tmp.find(x)
                if idx != -1:
                    #if this is just one word per line - make sure we don't blow past the boundaries
                    if idx == 0: #found at the beginning
                        tmp = str(tmp[idx + len(x):])
                    else:
                        tmp = str(tmp[:idx - 1] + tmp[idx + len(x):])
            keywords.extend(emails)

    if is_phone_num(tmp):
        matches = regex_phone.findall(tmp)
        if len(matches) > 0:
            phone_num = []
            [phone_num.append(val[0]) for count, val in enumerate(matches)]
            for c,x in enumerate(phone_num):
                idx = tmp.find(x)
                if idx != -1:
                    #if this is just one word per line - make sure we don't blow past the boundaries
                    if idx == 0: #found at the beginning
                        tmp = str(tmp[idx + len(x):])
                    else:
                        tmp = str(tmp[:idx - 1] + tmp[idx + len(x):])
            keywords.extend(list(map(lambda x: re.sub('[-\.\s]', '', x), phone_num)))

    if is_web_address(tmp):
        matches = regex_web.findall(tmp)
        if len(matches) > 0:
            [keywords.append(val[0]) for count, val in enumerate(matches)]
            for c, x in enumerate(keywords):
                idx = tmp.find(x)
                if idx != -1:
                    #if this is just one word per line - make sure we don't blow past the boundaries
                    if idx == 0: #found at the beginning
                        tmp = str(tmp[idx + len(x):])
                    else:
                        tmp = str(tmp[:idx - 1] + tmp[idx + len(x):])
    the_rest = str(tmp)
    return keywords, the_rest

def process_txt(txt):
    flag = False
    try:
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        s = ''
        if len(txt) > 0:
            txt = txt.lower()
            txt = re.sub('[^a-zA-Z\d()]', ' ', txt)
            txt = re.sub('[\s:]', ' ', txt)
            txt = re.sub('\s{2,}', ' ', txt)

            if txt.find(' ') == 0:
                txt = txt[1:]
            if txt.rfind(' ') == len(txt) - 1:
                txt = txt[:-1]
            keywords, the_rest = get_keywords(txt)
            if len(keywords):
                for keyword in keywords:
                    s += keyword
                    s += ' '
            words = nltk.word_tokenize(the_rest)
            for word in words:
                if len(word) < 3:
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
        vocab_path = join(dir, d1)
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
        txtfilepath = join(vocab_path, 'Respicio-pp.txt')
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
                    keywords, the_rest = get_keywords(line)
                    if len(keywords):
                        for keyword in keywords:
                            raw.append(keyword)
                    flag, txt = process_txt(the_rest)
                    if flag:
                        if len(txt) > 2:
                            raw.append(txt)
                if len(raw) > 0:
                    for sentence in raw:
                        words = []
                        keywords, the_rest = get_keywords(sentence)
                        if len(keywords):
                            words.extend(keywords)
                        if len(the_rest) > 0:
                            words.extend(nltk.word_tokenize(the_rest))
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

def process_raw(v):
    clean_up()
    process_inputs()
    process_vocab()

def main():
    args = os.argv[1:]
    v = args if len(args) > 0 else ''
    process_raw(v)
if __name__ == "__main__":
    main()