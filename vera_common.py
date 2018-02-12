import os
import logging as log

class Commons(object):
    def __init__(self):
        log.info('AI::VERA - Instance of %s has been created', type(self).__name__)
    @staticmethod
    def getEnv(name):
        val = ''
        try:
           val = os.environ[name]
           log.debug('AI::VERA - %s = %s.getEnv(%s)',val,\
                     Commons.__class__.__name__, name)
        except KeyError as e:
            log.error('AI::VERA - Exception %s caught in %s.getEnv(%s)', \
                      e.__class__.__name__, Commons.__class__.__name__, name)
        return val

    @staticmethod
    def isEnv(name):
        flag = False
        try:
            val = os.environ[name]
            if len(val) > 0:
                flag = True
            log.debug('AI::VERA - %s = %s.isEnv(%s)', val, \
                      Commons.__class__.__name__, name)
        except KeyError as e:
            flag = False
            log.error('AI::VERA - %s does not exist', name)
        return flag