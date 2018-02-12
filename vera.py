import os
from os import environ
import vera_setup
import threading
import time
from vera_processor import run_img
from vera_processor import run_raw
from vera_processor import run_input
from vera_common import Commons
import logging as log
import signal
from vera_spinner import Spinner

log.basicConfig(level=log.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class Config:
    vera = 'VERA_ROOT'
    debug = True
    wait_for_event_timeout = 5

def invalidPath(env, val):
    log.error('Error: environment variable %s is not set', env)
    if len(val) > 0:
        log.error('Path %s appears to be invalid. Re-initialize %s', val, env)

def checkPath(p, dir=False, debug=False):
    check = False
    if not os.path.exists(p):
        check = False
    if dir:
        check = os.path.isdir(p)
    else:
        check = os.path.isfile(p)
    log.debug('Path %s', ' does not exist ' if check == True else ' exists')
    return check

flag = True

def handle_keyboard_interrupt():
    log.info('AI::VERA - Keyboard interrupt caught')
    global flag
    flag = False

def run():
    global flag
    flag = True
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    log.info('AI::VERA - creating stopEvent')
    stopEvent = threading.Event()
    t = Config.wait_for_event_timeout
    log.info('AI::VERA - stopEvent timeout=%d', t)
    thx_img = 'run_image'
    t_img = threading.Thread(target=run_img, name=thx_img, args=(stopEvent,t,Config.debug,))
    log.info('AI::VERA - launching thread %s', thx_img)
    t_img.start()
    thx_raw = 'run_raw'
    t_raw = threading.Thread(target=run_raw, name=thx_raw, args=(stopEvent,t,Config.debug,))
    log.info('AI::VERA - launching thread %s', thx_raw)
    t_raw.start()
    thx_inp = 'run_input'
    t_inp = threading.Thread(target=run_input, name=thx_inp, args=(stopEvent,t,Config.debug,))
    log.info('AI::VERA - launching thread %s', thx_inp)
    t_inp.start()
    log.info('AI::VERA - running')
    while True:
        try:
            while flag:
                time.sleep(1)
            stopEvent.set()
            break
        except KeyboardInterrupt:
            log.info('AI::VERA - keyboard interrupt has been caught. Stopping...')
            stopEvent.set()
            break
    log.info('AI::VERA - joining on %s', t_img.getName())
    t_img.join()
    log.info('AI::VERA - joining on %s', t_raw.getName())
    t_raw.join()
    log.info('AI::VERA - joining on %s', t_inp.getName())
    t_inp.join()
    log.info('AI::VERA - has stopped running')

def main():
    level = log.DEBUG if Config.debug else log.INFO
    log.basicConfig(level=level)
    log.info('AI::VERA - Launching...')
    if Commons.isEnv(Config.vera):
        vera = Commons.getEnv(Config.vera)
        if len(vera) < 1 or checkPath(vera, dir=False):
            invalidPath(Config.vera, vera)
            return 0
        log.info('AI::VERA - is running file system check...')
        vera_setup.setup()
        run()
        log.info('AI::VERA - has terminated')

if __name__ == '__main__':
    main()