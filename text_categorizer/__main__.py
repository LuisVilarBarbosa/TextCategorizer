#!/usr/bin/python3
# coding=utf-8

import multiprocessing
import classifiers
import importlib.util
import pickle_manager

from os.path import isfile
from pandas import read_excel
from profilehooks import profile
from sys import argv
from feature_extraction import generate_X_y
from functions import data_frame_to_document_list
from Parameters import Parameters
from preprocessing import preprocess
from ui import verify_python_version

@profile
def main():
    if len(argv) != 2:
        print("Usage: python3 text_categorizer <configuration file>")
        quit()
    verify_python_version()
    config_filename = argv[1]
    Parameters.load_configuration(config_filename)
    numProcesses = Parameters.NUMBER_OF_PROCESSES
    if numProcesses == 0:
        numProcesses = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=numProcesses) as pool:
        if Parameters.PREPROCESS_DATA:
            if not isfile(Parameters.EXCEL_FILE) and not isfile(Parameters.PREPROCESSED_DATA_FILE):
                print("Please, provide a valid Excel file or a valid preprocessed data file.")
                quit()
            if not isfile(Parameters.PREPROCESSED_DATA_FILE) and isfile(Parameters.EXCEL_FILE):
                print("Loading Excel file.")
                data_frame = read_excel(Parameters.EXCEL_FILE)
                print("Executing initial_code_to_run_on_data_frame().")
                spec = importlib.util.spec_from_file_location("excel_filtration_code", Parameters.EXCEL_FILTRATION_CODE)
                excel_filtration_code = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(excel_filtration_code)
                data_frame = excel_filtration_code.initial_code_to_run_on_data_frame(data_frame)
                print("Creating documents.")
                docs = data_frame_to_document_list(data_frame)
                print("Storing generated documents.")
                pickle_manager.dump_documents(docs)
            print("Preprocessing documents.")
            preprocess()
            print("Checking generated data.")
            pickle_manager.check_data()
        else:
            if not isfile(Parameters.PREPROCESSED_DATA_FILE):
                print("The indicated preprocessed data file does not exist.")
                quit()
        print("Extracting features.")
        X, y = generate_X_y()
        print("Running classifiers.")
        print("Accuracies:")
        clfs = [
            classifiers.RandomForestClassifier,
            classifiers.BernoulliNB,
            classifiers.GaussianNB,
            classifiers.MultinomialNB,
            classifiers.ComplementNB,
            classifiers.KNeighborsClassifier,
            #classifiers.BernoulliRBM,
            classifiers.MLPClassifier,
            classifiers.LinearSVC,
            #classifiers.NuSVC,
            classifiers.DecisionTreeClassifier,
            classifiers.ExtraTreeClassifier,
            #classifiers.ClassifierMixin,
            classifiers.DummyClassifier,
        ]
        p = classifiers.Pipeline(clfs)
        p.start(X, y)

if __name__ == "__main__":
    main()
