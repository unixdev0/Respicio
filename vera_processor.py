import time
import logging as log
from watchdog.observers import Observer
import os
from os import environ
from vera_config import Config
import vera_handler
from vera_common import Commons
from vera_config import Setup

def run_img(stopEvent, timeout, debug=False):
    path = os.path.join(Commons.getEnv(Config.vera), Setup.path_img)
    event_handler = vera_handler.ImageHandler(debug)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    log.info('AI::VERA - running <%s> Watching path: [%s]', 'run_img', path)
    try:
        while True:
            event_set = stopEvent.wait(timeout)
            if event_set:
                break
            else:
                continue
    except:
        observer.stop()
    log.info('AI::VERA - stopping %s', 'run_img')
    if observer.isAlive():
        observer.stop()
    observer.join()
    return True

def run_raw(stopEvent, timeout, debug=False):
    path = os.path.join(Commons.getEnv(Config.vera), Setup.path_raw)
    event_handler = vera_handler.RawHandler(debug)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    log.info('AI::VERA - running <%s> Watching path: [%s]', 'run_raw', path)
    try:
        while True:
            event_set = stopEvent.wait(timeout)
            if event_set:
                break
            else:
                continue
    except:
        observer.stop()
    log.info('AI::VERA - stopping %s', 'run_raw')
    if observer.isAlive():
        observer.stop()
    observer.join()
    return True

def run_input(stopEvent, timeout, debug=False):
    path = os.path.join(Commons.getEnv(Config.vera), Setup.path_inp)
    event_handler = vera_handler.InputHandler(debug)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    log.info('AI::VERA - running <%s> Watching path: [%s]', 'run_img', path)
    try:
        while True:
            event_set = stopEvent.wait(timeout)
            if event_set:
                break
            else:
                continue
    except:
        observer.stop()
    log.info('AI::VERA - stopping %s', 'run_img')
    if observer.isAlive():
        observer.stop()
    observer.join()
    return True