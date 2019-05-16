#!/usr/bin/python3
# coding=utf-8

import pickle_manager

from numpy import arange
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm import tqdm
from logger import logger

def train_test_split(classifications, test_size, preprocessed_data_file):
    metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
    projected_test_size = metadata.get('test_size')
    if projected_test_size is not None and projected_test_size == test_size:
        logger.info("Using training and test subsets chosen in a previous execution.")
        return
    logger.info("Generating new training and test subsets.")
    metadata['test_size'] = test_size
    indexes = arange(len(classifications))
    training_set_indexes, test_set_indexes = sk_train_test_split(indexes, test_size=test_size,  random_state=42, shuffle=True, stratify=classifications)
    metadata['training_set_indexes'] = training_set_indexes
    metadata['test_set_indexes'] = test_set_indexes
    docs = pickle_manager.get_documents(preprocessed_data_file)
    total = metadata['total']
    pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
    for doc in tqdm(iterable=docs, desc="Storing subsets", total=total, unit="doc", dynamic_ncols=True):
        pda.dump_append(doc)
    pda.close()
