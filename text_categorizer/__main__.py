#!/usr/bin/python3
# coding=utf-8

import multiprocessing
import numpy
import text_categorizer.parameters as parameters

from os.path import isfile
from pandas import read_excel
from profilehooks import profile
from sys import argv
from text_categorizer.preprocessing import preprocess
from text_categorizer.ui import verify_python_version

@profile
def main():
    if len(argv) != 2:
        print("Usage: python3 text_categorizer <Excel file>")
        quit()
    verify_python_version()
    excel_file = argv[1]
    if not isfile(excel_file):
        print("The indicated Excel file does not exist.")
    else:
        print("Loading Excel file.")
        data_frame = read_excel(excel_file)
        numProcesses = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=numProcesses) as pool:
            print("Preprocessing data.")
            data = data_frame[parameters.EXCEL_COLUMN_WITH_TEXT_DATA]
            pool.map(preprocess, numpy.array_split(data, numProcesses))

if __name__ == "__main__":
    main()
