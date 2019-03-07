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

def data_frame_to_document_list(data_frame):
    from Document import Document
    documents = []
    for i in range(len(data_frame)):
        d = Document(data_frame=data_frame, index=i)
        documents.append(d)
    return documents
