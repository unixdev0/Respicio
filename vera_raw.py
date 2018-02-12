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

import logging as log

class ProcessRaw(object):
    regex_email = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    regex_web = re.compile(r"(([a-z]{1,3}\.)?([a-z]+\.[a-z]{1,3}))")
    regex_phone = re.compile(r"\s(((\+?1[-\.\s]?)?(\d{3}[-\.\s]?))(\d{3}[-\.\s]?\d{2}[-\.\s]?\d{2}))\s?")
    # regex_url   = re.compile(r"(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

    def __init__(self):
        log.info('AI::VERA - Instance of %s has been created', type(self).__name__)

    def is_email(self, word):
        is_match = self.regex_email.search(word)
        if not is_match is None:
            someFlag = True
        return not (is_match is None)

    def is_phone_num(self, word):
        is_match = self.regex_phone.search(word)
        return not (is_match is None)

    def is_web_address(self, word):
        is_match = self.regex_web.search(word)
        return not (is_match is None)

    def is_key_word(self, word):
        flag = False
        if self.is_email(word):
            flag = True
        elif self.is_phone_num(word):
            flag = True
        elif self.is_web_address(word):
            flag = True
        return flag

    def get_keywords(self, txt):
        keywords = []
        matches = []
        tmp = str(txt)
        the_rest = str(txt)

        if self.is_email(tmp):
            matches = self.regex_email.findall(tmp)
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

        if self.is_phone_num(tmp):
            matches = self.regex_phone.findall(tmp)
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

        if self.is_web_address(tmp):
            matches = self.regex_web.findall(tmp)
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

    def process_txt(self, txt):
        flag = False
        try:
            tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
            s = ''
            if len(txt) > 0:
                txt = txt.lower()
                txt = re.sub('[^a-zA-Z\d()]', ' ', txt)
                txt = re.sub('[\s:]', ' ', txt)
                txt = re.sub('\s{2,}', ' ', txt)
                '''
                if txt.find(' ') == 0:
                    txt = txt[1:]
                if txt.rfind(' ') == len(txt) - 1:
                    txt = txt[:-1]
                '''
                txt = txt.strip()
                keywords, the_rest = self.get_keywords(txt)
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
                '''if s.rfind(' ') == len(s) - 1:
                    s = s[:-1]'''
                s = s.rstrip()

            flag = True
        except:
            flag = False
        return flag, s

    def process_vocab(self, vera_path):
        #Accerss all the vera.ai files under the given path and generate the superset - vocab
        vocab_path = join(vera_path, 'vocab.vera')
        vocab = set([])
        for f in glob(path.join(vera_path, '*-vera.ai')):
            #for each vera.ai file create a set of words
            lines = open(f).readlines()
            if len(lines) > 0:
                vocab = vocab | set(lines)
        if path.exists(vocab_path):
            head, tail = path.split(vocab_path)
            tail = tail.replace(tail.rfind('.'), '-backup.')
            if path.exists(path.join(head, tail)):
                os.unlink(path.join(head, tail))
            os.rename(vocab_path, path.join(head, tail))
        hf = open(vocab_path, 'w')
        for item in vocab:
            hf.write(item)
        hf.close()

    def process_raw(self, ai_path, vera_path):
        d = set([])
        if path.isfile(ai_path) == True:
            ai_file = open(ai_path).readlines()
            raw = list()
            for line in ai_file:
                keywords, the_rest = self.get_keywords(line)
                if len(keywords):
                    for keyword in keywords:
                        raw.append(keyword)
                flag, txt = self.process_txt(the_rest)
                if flag:
                    if len(txt) > 2:
                        raw.append(txt)
            if len(raw) > 0:
                for sentence in raw:
                    words = []
                    keywords, the_rest = self.get_keywords(sentence)
                    if len(keywords):
                        words.extend(keywords)
                    if len(the_rest) > 0:
                        words.extend(nltk.word_tokenize(the_rest))
                    d = d | set(words)
        sd = set(sorted(d))
        vocab = set([])
        porter = nltk.PorterStemmer()
        for stemming in sd:
            if not self.is_key_word(stemming):
                stemmed_word = porter.stem(stemming)
            else:
                stemmed_word = stemming
            vocab.add(stemmed_word)
        vera_f = open(vera_path, 'w+')
        for item in vocab:
            vera_f.write(item)
            vera_f.write('\n')
        vera_f.close()
