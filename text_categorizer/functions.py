#!/usr/bin/python3
# coding=utf-8

from importlib.util import spec_from_file_location, module_from_spec
from os import path
from sys import version
from text_categorizer.Document import Document

def get_python_version():
    version_array = [int(n) for n in version[:version.find(" ")].split(".")]
    return version_array

def append_to_data_frame(array_2d, data_frame, column_name):
    new_data_frame = data_frame.copy()
    idx = len(new_data_frame.columns)
    new_column = []
    for array_1d in array_2d:
        new_column.append(','.join(array_1d))
    new_data_frame.insert(loc=idx, column=column_name, value=new_column, allow_duplicates=False)
    return new_data_frame

def data_frame_to_document_list(data_frame):
    documents = []
    for i in range(len(data_frame)):
        d = Document.from_data_frame(data_frame=data_frame, index=i)
        documents.append(d)
    return documents

def load_module(filename):
    name = path.splitext(path.basename(filename))[0]
    spec = spec_from_file_location(name, filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
