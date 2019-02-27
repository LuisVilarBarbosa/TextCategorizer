#!/usr/bin/python3
# coding=utf-8

def get_python_version():
    from sys import version
    version_number = int(version[:version.find(".")])
    return version_number

def append_to_data_frame(array_2d, data_frame, column_name):
    new_data_frame = data_frame.copy()
    idx = len(new_data_frame.columns)
    new_column = []
    for array_1d in array_2d:
        new_column.append(','.join([elem for elem in array_1d]))
    new_data_frame.insert(loc=idx, column=column_name, value=new_column, allow_duplicates=False)
    return new_data_frame
