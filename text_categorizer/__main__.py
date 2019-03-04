#!/usr/bin/python3
# coding=utf-8

import multiprocessing
import parameters

from os.path import isfile
from pandas import read_excel
from profilehooks import profile
from sys import argv
from classifiers import random_forest_classifier
from preprocessing import preprocess
from ui import verify_python_version

@profile
def main():
    if len(argv) != 3:
        print("Usage: python3 text_categorizer <Excel file> <Number of processes>")
        print(" - To use all machine cores, enter 0 in <Number of processes>.")
        quit()
    verify_python_version()
    excel_file = argv[1]
    numProcesses = argv[2]
    if not isfile(excel_file):
        print("The indicated Excel file does not exist.")
    elif not numProcesses.isnumeric():  # Checks if the input is a non-negative number.
        print("Invalid input for number of processes.")
    else:
        print("Loading Excel file.")
        data_frame = read_excel(excel_file)
        numProcesses = int(numProcesses)
        if numProcesses == 0:
            numProcesses = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=numProcesses) as pool:
            print("Preprocessing data.")
            texts = data_frame[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
            classifications = data_frame[parameters.EXCEL_COLUMN_WITH_CLASSIFICATION_DATA]
            texts = parameters.initial_code_to_run_on_text_data(texts)
            classifications = parameters.initial_code_to_run_on_classification_data(classifications)
            docs = preprocess(texts)
            print("Running classifier.")
            accuracy = random_forest_classifier(docs, classifications)
            print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
