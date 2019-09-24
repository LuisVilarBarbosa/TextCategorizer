import logging
from os.path import split
from text_categorizer import logger as logger_py
from text_categorizer.logger import logger
from tests.utils import default_encoding

def test_logger():
    assert logger.name == logger_py.package
    assert logger.propagate == False
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2
    assert all(type(handler) in [logging.FileHandler, logging.StreamHandler] for handler in logger.handlers)
    assert split(logger_py.file_handler.baseFilename)[1] == logger_py.filename
    assert logger_py.file_handler.mode == 'a'
    assert logger_py.file_handler.encoding == default_encoding
    assert logger_py.file_handler.level == logging.NOTSET
    assert logger_py.console_handler.level == logging.INFO
