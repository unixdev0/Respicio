import os
from os import path
from os import environ
import time
import logging
from os.path import join
from vera_config import Config
import logging as log

def checkDirPath(p, debug=False):
    log.debug('AI::VERA - checkDirPath %s', p)
    if not path.exists(p):
        log.debug('AI::VERA - Path %s does not exist', p)
        os.makedirs(p)
        log.debug('AI::VERA - Path %s has been created', p)
    else:
        log.debug('AI::VERA - Path %s already exists', p)


def setup():
    base_loc = environ[Config.vera]
    if len(base_loc) < 1:
        log.error('AI::VERA environment variable %s is not set', Config.vera)
        return False
    dirs = [('image', 'processed'), ('input', 'processed'), ('raw', 'processed'), ('template', 'backup'), ('model', 'backup'), 'train', 'tmp']
    for d in dirs:
        if isinstance(d, tuple):
            #TODO: fix this - should be able to build hierarchy programmanticall
            my_path = join(base_loc, d[0]) # @FIXME
            checkDirPath(my_path)
            my_path = join(my_path, d[1]) # @FIXME
            checkDirPath(my_path)
        else:
            checkDirPath(join(base_loc, d))
    return True

if __name__ == "__main__":
    setup(Config.debug)