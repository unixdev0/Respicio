from __future__ import division
import sys
from sys import path
import os
import os.path as path
import nltk, re, pprint

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
    dir = 'C:/Workspace/Bills'
    files = [
        'Doc-Respicio-001',
        'Doc-Respicio-002',
        'Doc-Respicio-003',
        'Doc-Respicio-004'
    ]
    ext = 'jpg'

    for f in files:
        filepath = dir + '/' + f + '.' + ext
        if path.isfile(filepath) == True:
            img = Image.open(filepath)
            txt = pytesseract.image_to_string(img)
            txtfilepath = dir + '/' + f + '.txt'
            flag,raw = process_txt(txt)
            if flag:
                txtf = open(txtfilepath, 'w')
                for i in raw:
                    txtf.write(i)
                    txtf.write('\n')
                txtf.close()
    err = 0

if __name__ == "__main__":
    run()