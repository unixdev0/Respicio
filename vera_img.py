from __future__ import division
import sys
from sys import path
import os
from os import environ
import os.path as path
import nltk, re, pprint
import logging as log

try:
    import Image
except ImportError:
    from PIL import Image
    import pytesseract

class ProcessImage(object):
    def __init__(self):
        log.info('AI::VERA - Instance %s has been created', type(self).__name__)
        pass

    def process_txt(self, txt):
        log.info('AI::VERA - %s.process_txt len(txt)=%d', type(self).__name__, len(txt))
        flag = False
        raw = list()
        try:
            if len(txt) > 0:
                txt = txt.lower()
                lines = re.split('\n', txt)
                for l in lines:
                    if l.isspace():
                        continue
                    if l.isdigit():
                        continue
                    if len(l) < 2:
                        continue
                    raw.append(l)
            flag = True
        except:
            flag = False
            log.error('AI::VERA - Exception caught in %s.process_txt', type(self).__name__)
        return flag,raw

    def process_img(self, path_img, path_raw):
        log.info('AI::VERA - %s.process_img path=%s', type(self).__name__, path_img)
        if path.isfile(path_img) == True:
            img = Image.open(path_img)
            log.info('AI::VERA - Running OCR on %s', path_img)
            txt = pytesseract.image_to_string(img)
            log.info('AI::VERA - OCR result len(txt)=%s', len(txt))
            flag,raw = self.process_txt(txt)
            if flag:
                raw_f = open(path_raw, 'w+')
                for i in raw:
                    raw_f.write(i)
                    raw_f.write('\n')
                raw_f.close()
                log.info('AI::VERA - raw text file %s has been created', path_raw)
            else:
                log.error('AI::VERA - failed to save raw text file %s', path_raw)
