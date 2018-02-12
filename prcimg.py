from __future__ import division
import sys
from sys import path
import os
import os.path as path
import nltk, re, pprint
from glob import glob
from os.path import join

try:
    import Image
except ImportError:
    from PIL import Image
    import pytesseract

def process_txt(txt):
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
    return flag,raw

def run():
    ext = '*.jpg'
    dir = 'C:/Workspace/Bills'
    ext_txt = 'txt'
    files =  glob(join(dir, 'image', ext))
    txtdir = path.join(dir, 'input')
    for f in files:
        if path.isfile(f) == True:
            img = Image.open(f)
            txt = pytesseract.image_to_string(img)
            head, tail = path.split(f)
            find_idx = tail.rfind('.jpg')
            new_tail = tail
            if find_idx != -1:
                new_tail = tail.replace('.jpg', '-raw.txt')
            txtfilepath = path.join(txtdir, new_tail)
            flag,raw = process_txt(txt)
            if flag:
                txtf = open(txtfilepath, 'w+')
                for i in raw:
                    txtf.write(i)
                    txtf.write('\n')
                txtf.close()
    err = 0

if __name__ == "__main__":
    run()