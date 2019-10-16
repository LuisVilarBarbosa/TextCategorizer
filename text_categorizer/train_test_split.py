#!/usr/bin/python3
# coding=utf-8

from collections import Counter
from numpy import arange
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.utils import safe_indexing
from text_categorizer import pickle_manager
from text_categorizer.logger import logger
from text_categorizer.ui import get_documents

def train_test_split(corpus, classifications, test_size, preprocessed_data_file, force, indexes_to_remove):
    metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
    projected_test_size = metadata.get('test_size')
    training_set_indexes = metadata.get('training_set_indexes')
    test_set_indexes = metadata.get('test_set_indexes')
    perform_split = force or projected_test_size is None or training_set_indexes is None or test_set_indexes is None
    if perform_split:
        logger.info("Generating new training and test subsets.")
        m = _train_test_split(metadata, test_size, classifications, indexes_to_remove)
        pickle_manager.set_docs_metadata(metadata=m, filename=preprocessed_data_file)
        metadata = pickle_manager.get_docs_metadata(preprocessed_data_file)
        projected_test_size = metadata.get('test_size')
        training_set_indexes = metadata.get('training_set_indexes')
        test_set_indexes = metadata.get('test_set_indexes')
    else:
        logger.info("Using training and test subsets chosen in a previous execution.")
        if projected_test_size != test_size or len(training_set_indexes) + len(test_set_indexes) != len(classifications) - len(indexes_to_remove):
            actual_test_size = len(test_set_indexes) / (len(classifications) - len(indexes_to_remove))
            logger.warning("The test subset corresponds to %s%% of the dataset instead of %s%%. The regeneration of the subsets can be enabled in the configuration file." % (actual_test_size, test_size))
        if not _is_stratified(classifications, metadata, indexes_to_remove):
            logger.warning("The training and test subsets are not correctly stratified. Are you using the correct classification column and ignoring the same examples? The regeneration of the subsets can be enabled in the configuration file.")
    return get_train_test(corpus, classifications, training_set_indexes, test_set_indexes, indexes_to_remove)

def get_train_test(corpus, classifications, training_set_indexes, test_set_indexes, indexes_to_remove):
    assert len(training_set_indexes) == len(set(training_set_indexes))
    assert len(test_set_indexes) == len(set(test_set_indexes))
    train_idxs = [i for i in training_set_indexes if not i in indexes_to_remove]
    test_idxs = [i for i in test_set_indexes if not i in indexes_to_remove]
    corpus_train = safe_indexing(corpus, train_idxs)
    corpus_test = safe_indexing(corpus, test_idxs)
    classifications_train = safe_indexing(classifications, train_idxs)
    classifications_test = safe_indexing(classifications, test_idxs)
    return corpus_train, corpus_test, classifications_train, classifications_test

def _train_test_split(metadata, test_size, classifications, indexes_to_remove):
    m = metadata.copy()
    m['test_size'] = test_size  
    idxs = arange(len(classifications))
    idxs = [i for i in idxs if not i in indexes_to_remove]
    class_labels = [classifications[i] for i in range(len(classifications)) if not i in indexes_to_remove]
    train_idxs, test_idxs = sk_train_test_split(idxs, test_size=test_size, random_state=42, shuffle=True, stratify=class_labels)
    m['training_set_indexes'] = train_idxs
    m['test_set_indexes'] = test_idxs
    return m

def _is_stratified(classifications, metadata, indexes_to_remove):
    train_labels = [classifications[i] for i in metadata['training_set_indexes']]
    test_labels = [classifications[i] for i in metadata['test_set_indexes']]
    class_labels = [classifications[i] for i in range(len(classifications)) if not i in indexes_to_remove]
    actual_test_size = len(test_labels) / len(class_labels)
    m = _train_test_split(metadata, actual_test_size, classifications, indexes_to_remove)
    expected_train_labels = [classifications[i] for i in m['training_set_indexes']]
    expected_test_labels = [classifications[i] for i in m['test_set_indexes']]
    return Counter(train_labels) == Counter(expected_train_labels) and Counter(test_labels) == Counter(expected_test_labels)
