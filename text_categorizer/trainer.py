#!/usr/bin/python3
# coding=utf-8

import pandas as pd
from copy import deepcopy
from os.path import exists, isfile
#from profilehooks import profile
from sklearn.datasets import fetch_20newsgroups
from text_categorizer import classifiers
from text_categorizer import functions, pickle_manager
from text_categorizer.constants import random_state
from text_categorizer.FeatureExtractor import FeatureExtractor
from text_categorizer.logger import logger
from text_categorizer.Parameters import Parameters
from text_categorizer.Preprocessor import Preprocessor
from text_categorizer.resampling import RandomOverSample, RandomUnderSample
from text_categorizer.train_test_split import train_test_split

def load_20newsgroups(parameters):
    p = deepcopy(parameters)
    p.excel_file = ''.join([p.excel_file, '.xlsx'])
    p.excel_column_with_text_data = 'data'
    p.excel_column_with_classification_data = 'target'
    if not exists(p.excel_file) and not exists(p.preprocessed_data_file):
        bunch = fetch_20newsgroups(data_home='.', subset='all', categories=None, shuffle=False, random_state=random_state, remove=(), download_if_missing=True)
        df = pd.DataFrame()
        df[p.excel_column_with_text_data] = bunch.data
        df[p.excel_column_with_classification_data] = bunch.target
        df.to_excel(p.excel_file, engine='xlsxwriter')
    return p

def resample(resampling, X_train, y_train):
    if resampling is None:
        return X_train, y_train
    elif resampling == RandomOverSample.__name__:
        logger.info("Starting random over sampler.")
        return RandomOverSample(X_train, y_train)
    elif resampling == RandomUnderSample.__name__:
        logger.info("Starting random under sampler.")
        return RandomUnderSample(X_train, y_train)
    else:
        error_msg = "Invalid resampling method."
        logger.error(error_msg)
        raise ValueError(error_msg)

#@profile
def main(config_filename):
    execution_info = pd.DataFrame()
    execution_info['Start'] = [functions.get_local_time_str()]
    logger.debug("Starting execution.")
    parameters = Parameters(config_filename)
    if parameters.excel_file == '20newsgroups':
        parameters = load_20newsgroups(parameters)
    if parameters.preprocess_data:
        if not isfile(parameters.excel_file) and not isfile(parameters.preprocessed_data_file):
            logger.error("Please, provide a valid Excel file or a valid preprocessed data file.")
            quit()
        if not isfile(parameters.preprocessed_data_file) and isfile(parameters.excel_file):
            logger.info("Loading Excel file.")
            data_frame = pd.read_excel(parameters.excel_file)
            data_frame = data_frame.fillna("NaN")
            logger.info("Creating documents.")
            docs = functions.data_frame_to_document_list(data_frame)
            logger.info("Storing generated documents.")
            pickle_manager.dump_documents(docs, parameters.preprocessed_data_file)
        logger.info("Preprocessing documents.")
        preprocessor = Preprocessor(stanfordnlp_language_package=parameters.stanfordnlp_language_package, stanfordnlp_use_gpu=parameters.stanfordnlp_use_gpu, stanfordnlp_resources_dir=parameters.stanfordnlp_resources_dir, store_data=True, spell_checker_lang=parameters.spell_checker_lang)
        preprocessor.preprocess(text_field=parameters.excel_column_with_text_data, preprocessed_data_file=parameters.preprocessed_data_file)
        logger.info("Checking generated data.")
        pickle_manager.check_data(parameters.preprocessed_data_file)
    else:
        if not isfile(parameters.preprocessed_data_file):
            logger.error("The indicated preprocessed data file does not exist.")
            quit()
    logger.info("Extracting features and splitting dataset into training and test subsets.")
    feature_extractor = FeatureExtractor(nltk_stop_words_package=parameters.nltk_stop_words_package, vectorizer_name=parameters.vectorizer, training_mode=True, feature_reduction=parameters.feature_reduction, document_adjustment_code=parameters.document_adjustment_code, remove_adjectives=parameters.remove_adjectives, synonyms_file=parameters.synonyms_file, vectorizer_file=parameters.vectorizer_file)
    corpus, classifications, idxs_to_remove, _docs_lemmas = feature_extractor.prepare(class_field=parameters.excel_column_with_classification_data, preprocessed_data_file=parameters.preprocessed_data_file)
    corpus_train, corpus_test, classifications_train, classifications_test = train_test_split(corpus, classifications, parameters.test_subset_size, parameters.preprocessed_data_file, parameters.force_subsets_regeneration, idxs_to_remove)
    X_train, y_train = feature_extractor.generate_X_y(corpus_train, classifications_train, training_mode=True)
    X_test, y_test = feature_extractor.generate_X_y(corpus_test, classifications_test, training_mode=False)
    X_train, y_train = resample(parameters.resampling, X_train, y_train)
    logger.info("Running classifiers.")
    p = classifiers.Pipeline(parameters.classifiers)
    logger.info("Accuracies:")
    predictions_dict = p.start(X_train, y_train, X_test, y_test, parameters.number_of_jobs, parameters.set_num_accepted_probs, parameters.class_weights, parameters.generate_roc_plots)
    execution_info['End'] = [functions.get_local_time_str()]
    logger.debug("Execution completed.")
    functions.generate_report(execution_info, parameters.__dict__, predictions_dict)
