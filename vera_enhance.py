from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from os import environ
from vera_config import Config
from os import path
from glob import glob
import json
import pprint
import os
import re
from vera_spinner import Spinner
import sys

def header():
    print '***************************************************************'
    print '*                                                             *'
    print '*                 R E S P I C . I O                           *'
    print '*                                                             *'
    print '*-------------------------------------------------------------*'


def error():
    print ' Error. No arguments have been provided. Exiting               '
    print '*-------------------------------------------------------------*'


def run():
    header()
    args = sys.argv[1:]
    if len(args) < 1:
        error()
        return False
    spinner = Spinner()
    enhancement_factor = 3
    vera = environ[Config.vera]
    for arg in args:
        if not path.exists(path.abspath(arg)):
            print 'Error: path ', arg, ' does not exist. Skipping...'
            continue
        for file in glob(path.join(path.abspath(arg), '*.tiff')):
            if '-enhanced' in file: continue
            print 'Processing file:', file
            spinner.start()
            io = Image.open(file)
            head, tail = path.split(file)
            idx = tail.rfind('.')
            cp_file_name = path.join(head, tail[:idx] + '-enhanced.' + tail[idx + 1:])
            io = io.filter(ImageFilter.MedianFilter())
            enhancer = ImageEnhance.Contrast(io)
            io = enhancer.enhance(enhancement_factor)
            brightness = ImageEnhance.Brightness(io)
            brightness.enhance(enhancement_factor)
            sharpness = ImageEnhance.Sharpness(io)
            sharpness.enhance(enhancement_factor)
            io = io.convert('1')
            if path.isfile(cp_file_name):
                try:
                    os.unlink(cp_file_name)
                except:
                    pass
            io.save(cp_file_name)
            spinner.stop()
            print 'Saved enhanced image copy as', cp_file_name

if __name__ == '__main__':
    run()

