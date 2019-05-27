#!/usr/bin/python3
# coding=utf-8

import classifiers
import pickle_manager

from os.path import isfile
from pandas import read_excel
#from profilehooks import profile
from sys import argv
from FeatureExtractor import FeatureExtractor
from functions import data_frame_to_document_list
from logger import logger
from Parameters import Parameters
from Preprocessor import Preprocessor
from train_test_split import train_test_split
from ui import verify_python_version

#@profile
def main():
    if len(argv) != 2:
        print("Usage: python3 text_categorizer <configuration file>")
        quit()
    logger.debug("Starting execution.")
    verify_python_version()
    config_filename = argv[1]
    parameters = Parameters(config_filename, training_mode=True)
    if parameters.preprocessed_data:
        if not isfile(parameters.excel_file) and not isfile(parameters.preprocessed_data_file):
            logger.error("Please, provide a valid Excel file or a valid preprocessed data file.")
            quit()
        if not isfile(parameters.preprocessed_data_file) and isfile(parameters.excel_file):
            logger.info("Loading Excel file.")
            data_frame = read_excel(parameters.excel_file)
            logger.info("Creating documents.")
            docs = data_frame_to_document_list(data_frame)
            logger.info("Storing generated documents.")
            pickle_manager.dump_documents(docs, parameters.preprocessed_data_file)
        logger.info("Preprocessing documents.")
        preprocessor = Preprocessor(parameters)
        preprocessor.preprocess()
        logger.info("Checking generated data.")
        pickle_manager.check_data(parameters.preprocessed_data_file)
    else:
        if not isfile(parameters.preprocessed_data_file):
            logger.error("The indicated preprocessed data file does not exist.")
            quit()
    logger.info("Extracting features.")
    feature_extractor = FeatureExtractor(nltk_stop_words_package=parameters.nltk_stop_words_package, vectorizer_name=parameters.vectorizer, training_mode=parameters.training_mode, use_lda=parameters.use_lda, document_adjustment_code=parameters.document_adjustment_code, remove_adjectives=parameters.remove_adjectives, synonyms_file=parameters.synonyms_file, features_file=parameters.features_file)
    X, y, _lemmas = feature_extractor.generate_X_y(class_field=parameters.excel_column_with_classification_data, preprocessed_data_file=parameters.preprocessed_data_file)
    logger.info("Splitting dataset into training and test subsets.")    
    train_test_split(y, parameters.test_subset_size, parameters.preprocessed_data_file, parameters.force_subsets_regeneration)
    logger.info("Running classifiers.")
    p = classifiers.Pipeline(parameters.classifiers, parameters.cross_validate)
    metadata = pickle_manager.get_docs_metadata(parameters.preprocessed_data_file)
    training_set_indexes = metadata['training_set_indexes']
    test_set_indexes = metadata['test_set_indexes']
    logger.info("Accuracies:")
    p.start(X, y, parameters.number_of_jobs, parameters.set_num_accepted_probs, training_set_indexes, test_set_indexes, parameters.resampling)
    logger.debug("Execution completed.")

if __name__ == "__main__":
    main()
