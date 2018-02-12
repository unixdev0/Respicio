from watchdog.events import PatternMatchingEventHandler
from vera_config import Config
import os
from os import path
from os.path import join
import logging as log
from vera_img import ProcessImage
from vera_raw import ProcessRaw
from vera_nn import Classifier
import vera_template as templates
import threading
import time
from vera_config import Setup
from vera_common import Commons

class ImageHandler(PatternMatchingEventHandler):
    patterns = ["*.jpg", "*.png", "*.tiff"]

    def __init__(self, debug):
        PatternMatchingEventHandler.__init__(self, patterns=self.patterns)
        self.debug = debug
        self.myPath = join(Commons.getEnv(Config.vera), Setup.path_img)
        self.nextPath = join(Commons.getEnv(Config.vera), Setup.path_raw)
        log.info('AI::VERA - Created instance: %s', type(self).__name__)

    def process(self, event):
        log.info('AI::VERA - %s:process type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        if event.event_type == 'modified' and not event.is_directory:
            head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                while True:
                    try:
                        os.remove(new_path)
                        break
                    except Exception as e:
                        log.error('AI::VERA - exception %s', e.message)
                        time.sleep(1)
                        continue
                    except:
                        log.error('AI::VERA - unknown exception caught')
                        time.sleep(1)
                        continue
            retry_delay = Setup.file_system_retry_delay
            for x in range(Setup.file_system_retries):
                try:
                    os.rename(event.src_path, new_path)
                    break
                except WindowsError as e:
                    log.error('AI::VERA - File system error [%s] File:%s Retrying[%d]', e.message, event.src_path, x)
                    time.sleep(retry_delay)
                    retry_delay *= 2
            head, tail = path.split(new_path)
            ext_idx = tail.rfind('.')
            if ext_idx != -1:
                tail = tail.replace(tail[ext_idx:], '-raw.ai')
            raw_file_path = join(self.nextPath, tail)
            processor = ProcessImage()
            processor.process_img(new_path, raw_file_path)
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
    def on_modified(self, event):
        log.info('AI::VERA - %s:on_modify type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        self.process(event)

class RawHandler(PatternMatchingEventHandler):
    patterns = ["*-raw.ai"]

    def __init__(self, debug):
        PatternMatchingEventHandler.__init__(self, patterns=self.patterns)
        self.debug = debug
        self.myPath = join(Commons.getEnv(Config.vera), Setup.path_raw)
        self.nextPath = join(Commons.getEnv(Config.vera), Setup.path_inp)
        log.info('AI::VERA - Created instance: %s', type(self).__name__)

    def process(self, event):
        log.info('AI::VERA - %s:process type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        if event.event_type == 'modified' and not event.is_directory:
            head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                os.remove(new_path)
            os.rename(event.src_path, new_path)
            head, tail = path.split(new_path)
            if tail.find('-raw.ai') != -1:
                tail = tail.replace('-raw.ai', '-vera.ai')
            vera_file_path = join(self.nextPath, tail)
            processor = ProcessRaw()
            processor.process_raw(new_path, vera_file_path)

    def on_modified(self, event):
        log.info('AI::VERA - %s:on_modify type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        self.process(event)

class InputHandler(PatternMatchingEventHandler):
    patterns = ["*-vera.ai"]

    def __init__(self, debug):
        PatternMatchingEventHandler.__init__(self, patterns=self.patterns)
        self.debug = debug
        self.classifier = Classifier()
        self.classifier.setup()
        log.info('AI::VERA - Created instance: %s', type(self).__name__)

    def process(self, event):
        log.info('AI::VERA - %s:process type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        if event.event_type == 'modified' and not event.is_directory:
            head, tail = path.split(event.src_path)
            new_path = join(head, 'processed', tail)
            if path.exists(new_path):
                os.remove(new_path)
            os.rename(event.src_path, new_path)
            pred = self.classifier.process(new_path)
            if len(pred):
                log.info('AI::VERA - %s:process document=%s has been classifed',\
                    type(self).__name__, event.src_path)
                for p in pred:
                    log.info('AI::VERA - Classification <<< %s >>>, Confidence [[[ %f ]]]', p[0].upper(), p[1])
                    t_templ = threading.Thread(target=templates.handle_TU, name='Data Extraction Thread', args=(p[0], new_path,))
                    t_templ.daemon = True
                    t_templ.start()
            else:
                head, tail = path.split(new_path)
                t = tail[:tail.rfind('-vera.ai')]
                log.error('AI::VERA - something went wrong and AI::VERA \
                was unable to classify the document %s', t)



    def on_modified(self, event):
        log.info('AI::VERA - %s:on_modify type=%s path=%s', type(self).__name__, event.event_type, event.src_path)
        self.process(event)
