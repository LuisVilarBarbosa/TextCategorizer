#!/usr/bin/python3
# coding=utf-8

import pickle_manager

from tqdm import tqdm

def verify_python_version():
    from functions import get_python_version
    from logger import logger
    version_array = get_python_version()
    if version_array < [3,5]:
        logger.error("Please use Python3.5 or higher.")
        quit()

def get_documents(filename, description=None):
    total = pickle_manager.get_docs_metadata(filename)['total']
    docs = pickle_manager.get_documents(filename)
    for doc in tqdm(iterable=docs, desc=description, total=total, unit="doc", dynamic_ncols=True):
        yield doc
