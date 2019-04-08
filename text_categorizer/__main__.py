#!/usr/bin/python3
# coding=utf-8

import classifiers
import importlib.util
import pickle_manager

from os.path import isfile
from pandas import read_excel
#from profilehooks import profile
from sys import argv
from feature_extraction import generate_X_y
from functions import data_frame_to_document_list
from logger import logger
from Parameters import Parameters
from preprocessing import preprocess
from ui import verify_python_version

#@profile
def main():
    if len(argv) != 2:
        print("Usage: python3 text_categorizer <configuration file>")
        quit()
    logger.debug("Starting execution.")
    verify_python_version()
    config_filename = argv[1]
    Parameters.load_configuration(config_filename, train_mode=True)
    if Parameters.PREPROCESS_DATA:
        if not isfile(Parameters.EXCEL_FILE) and not isfile(Parameters.PREPROCESSED_DATA_FILE):
            logger.error("Please, provide a valid Excel file or a valid preprocessed data file.")
            quit()
        if not isfile(Parameters.PREPROCESSED_DATA_FILE) and isfile(Parameters.EXCEL_FILE):
            logger.info("Loading Excel file.")
            data_frame = read_excel(Parameters.EXCEL_FILE)
            logger.info("Executing initial_code_to_run_on_data_frame().")
            spec = importlib.util.spec_from_file_location("excel_filtration_code", Parameters.EXCEL_FILTRATION_CODE)
            excel_filtration_code = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(excel_filtration_code)
            data_frame = excel_filtration_code.initial_code_to_run_on_data_frame(data_frame)
            logger.info("Creating documents.")
            docs = data_frame_to_document_list(data_frame)
            logger.info("Storing generated documents.")
            pickle_manager.dump_documents(docs)
        logger.info("Preprocessing documents.")
        preprocess()
        logger.info("Checking generated data.")
        pickle_manager.check_data()
    else:
        if not isfile(Parameters.PREPROCESSED_DATA_FILE):
            logger.error("The indicated preprocessed data file does not exist.")
            quit()
    logger.info("Extracting features.")
    X, y = generate_X_y()
    logger.info("Running classifiers.")
    clfs = [
        classifiers.RandomForestClassifier,
        classifiers.BernoulliNB,
        classifiers.MultinomialNB,
        classifiers.ComplementNB,
        classifiers.KNeighborsClassifier,
        classifiers.MLPClassifier,
        classifiers.SVC,
        classifiers.DecisionTreeClassifier,
        classifiers.ExtraTreeClassifier,
        classifiers.DummyClassifier,
        classifiers.SGDClassifier,
        classifiers.BaggingClassifier,
        classifiers.RFE,
        classifiers.RFECV,
    ]
    p = classifiers.Pipeline(clfs)
    logger.info("Accuracies:")
    p.start(X, y)
    logger.debug("Execution completed.")

if __name__ == "__main__":
    main()
