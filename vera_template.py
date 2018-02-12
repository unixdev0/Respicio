from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from os import environ
from vera_config import Config
from os import path
from glob import glob
import json
import logging as log
import pprint
import os
import re
from vera_common import Commons
from vera_config import Setup
import time


def validateCoordinates(tup):
    flag = False
    if len(tup) != 4:
        log.error('AI::VERA - Invalid tuple length %s', tup)
    elif min(tup) < 0:
        log.error('AI::VERA - Invalid tuple: negative coordinates X:Y are not allowed %s', tup)
    elif tup[2] < tup[0]:
        log.error('AI::VERA - Invalid tuple: upper left X coordinate %i is greater than lower left %i', tup[0], tup[2])
    elif tup[3] < tup[1]:
        log.error('AI::VERA - Invalid tuple: upper left Y coordinate %i is greater than lower left %i', tup[1], tup[3])
    else:
        flag = True
    return flag


def get_model_template():
    vera = Commons.getEnv(Config.vera)
    model_path = path.join(vera, Setup.path_model, Setup.model_config)
    log.info('AI::VERA - Loading model metadata for type <<< %s >>>', Setup.model.upper())
    if not path.exists(model_path):
        # @TODO: handle differently to indicate that initialization failed
        log.error('AI::VERA - Cannot find model metadata file %s', model_path)
        return
    meta = json.load(open(model_path, 'r'))
    template = ''
    for m in meta['models']:
        if Setup.model in m['name']:
            template = m['template']
            break
    return template


def handle_TU(name, tfile):
    enhancment_factor = 2
    vera = environ[Config.vera]
    log.info('AI::VERA - loading the templates')
    template_name = get_model_template()
    templates = path.join(vera, 'template', template_name)
    d = dict()
    base = re.sub(r'-vera.ai', Setup.image_ext, path.basename(tfile))
    tu_file = path.join(vera, 'image', 'processed', base)
    if not path.isfile(tu_file):
        log.error('AI::VERA - cannot locate image file %s. Bailing out', tu_file)
    if path.isfile(templates):
        t = open(templates).read()
        log.info('AI::VERA - successfully loaded the templates. Parsing...')
        j = json.loads(t)
        l = j['templates']
        for x in l:
            d[x['name']] = x['fields']
    log.info('AI::VERA - successfully parsed the templates')
    try:
        d[name]
    except KeyError:
        log.error('AI::VERA - template for type %s cannot be located. No further processing will be performed', name)
        return
    work_template = d[name]
    log.info('AI::VERA - Analyzing file %s', tu_file)
    #@TODO: fix missing files
    timeout = Setup.file_system_retry_delay
    file_flag = False
    for retry in range(Setup.file_system_retries):
        try:
            ii = Image.open(tu_file)
            file_flag = True
            break
        except WindowsError as we:
            log.info('AI::VERA - %s : %s', type(we).__name__, we.message)
            time.sleep(timeout)
            timeout *= 2
            continue
        except IOError as ioe:
            log.info('AI::VERA - %s : %s', type(we).__name__, we.message)
            time.sleep(timeout)
            timeout *= 2
            continue
    if not file_flag:
        log.error('AI::VERA - cannot find %s. Bailing out', tu_file)
        return
    head, tail = path.split(tu_file)
    idx = tail.rfind('.')
    cp_file_name = path.join(head, tail[:idx] + '-copy.' + tail[idx + 1:])
    log.info('AI::VERA - Creating secondary image file %s', cp_file_name)
    io = Image.new(ii.mode, (ii.width, ii.height), 'white')
    boxes = list()
    log.info('AI::VERA - Creating image cropping coordinates for %s', cp_file_name)
    log.info('AI::VERA - Parsing and applying text extraction template [%s]', work_template)
    for item in work_template:
        keys = item.keys()
        for key in keys:
            boxes.append(item[key])
    log.info('AI::VERA - Cropping secondary image file %s', cp_file_name)
    for box in boxes:
        t = tuple(box)
        if not validateCoordinates(t):
            log.error('AI::VERA - Invalid rectangle coordinates %s', t)
        region = ii.crop(t)
        if min(region.size) == 0:
            log.error('AI::VERA - Region (%s) has not been cropped correctly. Check the rectangle coordinates', t)
        io.paste(region, (t[0], t[1]))
    if Setup.image_enhancement:
        log.info('AI::VERA - Configuration indicates that image enhancements to be performed')
        log.info('AI::VERA - OCR Enhancing image file %s', cp_file_name)
        io = io.filter(ImageFilter.MedianFilter())
        log.info('AI::VERA - OCR Boosting image contract by factor %ix', enhancment_factor)
        enhancer = ImageEnhance.Contrast(io)
        io = enhancer.enhance(enhancment_factor)
        log.info('AI::VERA - OCR Boosting image brightness by factor %ix', enhancment_factor)
        brightness = ImageEnhance.Brightness(io)
        brightness.enhance(enhancment_factor)
        log.info('AI::VERA - OCR Boosting image sharpness by factor %ix', enhancment_factor)
        sharpness = ImageEnhance.Sharpness(io)
        sharpness.enhance(enhancment_factor)
        log.info('AI::VERA - running OCR image grey-scaling')
        io = io.convert('1')
        io.load()
    log.info('AI::VERA - running OCR text recognition on %s', cp_file_name)
    txt = pytesseract.image_to_string(io, config="-psm 6")
    log.info('AI::VERA - OCR text of %i characters has been extracted from %s', len(txt), cp_file_name)
    log.info('AI::VERA - showing OCR text >>>')
    col = txt.split('\n')
    pprint.pprint(col)
    log.info('AI::VERA - Saving image %s', cp_file_name)
    if path.isfile(cp_file_name):
        os.unlink(cp_file_name)
    io.save(cp_file_name)
    log.info('AI::VERA - DONE processing %s', cp_file_name)

if __name__ == '__main__':
    handle_TU()

