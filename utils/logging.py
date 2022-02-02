"""
@author: Sotiris
"""
import sys
import logging
import os
from typing import Union, TextIO


class LoggingConfig(object):
    __logger = logging.getLogger(__name__)
    root = None
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'error': logging.ERROR,
              'warning': logging.WARNING}

    HANDLERS = {'stream': logging.StreamHandler,
                'file': logging.FileHandler}

    def __init__(self, level, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handler_type='stream',
                 param: Union[str, TextIO, None] = sys.stdout):
        assert level in LoggingConfig.LEVELS
        assert handler_type in LoggingConfig.HANDLERS
        self.level = LoggingConfig.LEVELS[level]
        self.fmt = fmt
        self.handler_type = handler_type
        self.handler = LoggingConfig.HANDLERS[handler_type]
        self.param = param
        self.set_logger()

    def set_logger(self):
        self.root = logging.getLogger()
        self.root.setLevel(self.level)
        if self.handler_type == 'file':
            self.create_dirs_if_needed(self.param)
        handler = self.handler(self.param)
        handler.setLevel(self.level)
        formatter = logging.Formatter(self.fmt)
        handler.setFormatter(formatter)
        self.root.addHandler(handler)
        return self.root

    def get_logger(self):
        return self.root

    @staticmethod
    def create_dirs_if_needed(filepath):
        if '/' in filepath:
            path = '/'.join(filepath.split('/')[:-1])
            if not os.path.exists(path):
                os.makedirs(path)