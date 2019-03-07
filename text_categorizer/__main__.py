#!/usr/bin/python3
# coding=utf-8

import multiprocessing
import jsonpickle_manager
import parameters

from os.path import isfile
from pandas import read_excel
from profilehooks import profile
from sys import argv
from classifiers import random_forest_classifier
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
            if not isfile(parameters.EXCEL_FILE):
                print("The indicated Excel file does not exist.")
                quit()
            print("Loading Excel file.")
            data_frame = read_excel(parameters.EXCEL_FILE)
            print("Executing initial_code_to_run_on_data_frame().")
            data_frame = parameters.initial_code_to_run_on_data_frame(data_frame)
            print("Creating documents.")
            docs = data_frame_to_document_list(data_frame)
            print("Preprocessing documents.")
            docs = preprocess(docs)
            print("Storing preprocessed documents.")
            jsonpickle_manager.dump(obj=docs, filename=parameters.PREPROCESSED_DATA_FILE)
        else:
            print("Loading preprocessed documents.")
            docs = jsonpickle_manager.load(filename=parameters.PREPROCESSED_DATA_FILE)
        print("Extracting features.")
        X, y = generate_X_y(docs)
        print("Running classifier.")
        accuracy = random_forest_classifier(X, y)
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
