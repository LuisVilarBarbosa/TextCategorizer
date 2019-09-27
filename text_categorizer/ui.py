#!/usr/bin/python3
# coding=utf-8

from text_categorizer import pickle_manager
from text_categorizer.functions import get_python_version
from text_categorizer.logger import logger
from tqdm.autonotebook import tqdm

def verify_python_version():
    version_array = get_python_version()
    if version_array < [3,5]:
        logger.error("Please use Python3.5 or higher.")
        quit()

def get_documents(filename, description=None):
    total = pickle_manager.get_docs_metadata(filename)['total']
    docs = pickle_manager.get_documents(filename)
    for doc in progress(iterable=docs, desc=description, total=total, unit="doc", dynamic_ncols=True):
        yield doc

def progress(**kwargs):
    return tqdm(**kwargs)
