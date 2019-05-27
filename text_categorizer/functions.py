#!/usr/bin/python3
# coding=utf-8

def get_python_version():
    from sys import version
    version_array = [int(n) for n in version[:version.find(" ")].split(".")]
    return version_array

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
        d = Document.from_data_frame(data_frame=data_frame, index=i)
        documents.append(d)
    return documents

def load_module(filename):
    from os import path
    from importlib.util import spec_from_file_location, module_from_spec
    name = path.splitext(path.basename(filename))[0]
    spec = spec_from_file_location(name, filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
