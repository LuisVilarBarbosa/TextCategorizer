#!/usr/bin/python3
# coding=utf-8

import pickle_manager

from numpy import arange
from sklearn.model_selection import train_test_split as sk_train_test_split
from logger import logger
from ui import get_documents

def train_test_split(classifications, test_size, preprocessed_data_file, force):
    perform_split = False
    metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
    projected_test_size = metadata.get('test_size')
    if force or projected_test_size is None or projected_test_size != test_size:
        perform_split = True
    training_set_indexes = metadata['training_set_indexes']
    test_set_indexes = metadata['test_set_indexes']
    if len(training_set_indexes) + len(test_set_indexes) != len(classifications):
        perform_split = True
    if not perform_split:
        logger.info("Using training and test subsets chosen in a previous execution.")
        return
    logger.info("Generating new training and test subsets.")
    metadata['test_size'] = test_size
    indexes = arange(len(classifications))
    training_set_indexes, test_set_indexes = sk_train_test_split(indexes, test_size=test_size,  random_state=42, shuffle=True, stratify=classifications)
    metadata['training_set_indexes'] = training_set_indexes
    metadata['test_set_indexes'] = test_set_indexes
    pda = pickle_manager.PickleDumpAppend(metadata=metadata, filename=preprocessed_data_file)
    for doc in get_documents(preprocessed_data_file, description="Storing subsets"):
        pda.dump_append(doc)
    pda.close()
