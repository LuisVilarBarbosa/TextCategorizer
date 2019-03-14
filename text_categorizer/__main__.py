#!/usr/bin/python3
# coding=utf-8

import multiprocessing
import classifiers
import pickle_manager
import parameters

from os.path import isfile, isdir
from pandas import read_excel
from profilehooks import profile
from sys import argv
from feature_extraction import generate_X_y
from functions import data_frame_to_document_list
from preprocessing import preprocess
from ui import verify_python_version

@profile
def main():
    if len(argv) != 1:
        print("Usage: python3 text_categorizer")
        quit()
    verify_python_version()
    numProcesses = int(parameters.NUMBER_OF_PROCESSES)
    if numProcesses == 0:
        numProcesses = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=numProcesses) as pool:
        if parameters.PREPROCESS_DATA:
            if not isfile(parameters.EXCEL_FILE) and not isdir(parameters.PREPROCESSED_DATA_FOLDER):
                print("Please, provide a valid Excel file or a valid preprocessed data folder.")
                quit()
            if not isdir(parameters.PREPROCESSED_DATA_FOLDER) and isfile(parameters.EXCEL_FILE):
                print("Loading Excel file.")
                data_frame = read_excel(parameters.EXCEL_FILE)
                print("Executing initial_code_to_run_on_data_frame().")
                data_frame = parameters.initial_code_to_run_on_data_frame(data_frame)
                print("Creating documents.")
                docs = data_frame_to_document_list(data_frame)
                print("Storing generated documents.")
                pickle_manager.dump_all_documents(docs)
            print("Preprocessing documents.")
            preprocess()
        else:
            if not isdir(parameters.PREPROCESSED_DATA_FOLDER):
                print("The indicated preprocessed data folder does not exist.")
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
        ]
        p = classifiers.Pipeline(clfs)
        p.start(X, y)

if __name__ == "__main__":
    main()
