#!/usr/bin/python3
# coding=utf-8

import text_categorizer.parameters as parameters

from os.path import isfile
from pandas import read_excel
from sys import argv
from text_categorizer.preprocessing import preprocess
from text_categorizer.ui import verify_python_version

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
        print("Preprocessing data.")
        preprocess(data_frame[parameters.EXCEL_COLUMN_WITH_TEXT_DATA])

if __name__ == "__main__":
    main()
