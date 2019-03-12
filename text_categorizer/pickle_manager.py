#!/usr/bin/python3
# coding=utf-8

import pickle
import os
import parameters

from Document import Document

def dump(obj, path):
    output_file = open(path, 'wb')
    version = 2
    pickle.dump(obj, output_file, version)
    output_file.close()

def load(path):
    input_file = open(path, 'rb')
    data = pickle.load(input_file)
    input_file.close()
    return data

def load_all_documents():
    from numpy import append
    paths = files_paths()
    docs = []
    for path in paths:
        ds = load_documents(path)
        docs = append(docs, ds)
    docs = docs.flatten().tolist()
    docs.sort(key=lambda doc: doc.index)
    return docs

_metadata_file = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, "metadata.pkl")

def load_documents(path):
    input_file = open(path, 'rb')
    docs = []
    while True:
        try:
            docs.append(pickle.load(input_file))
        except EOFError:
            input_file.close()
            return docs

def dump_all_documents(docs):
    os.makedirs(parameters.PREPROCESSED_DATA_FOLDER, exist_ok=True)
    if len(os.listdir(parameters.PREPROCESSED_DATA_FOLDER)) > 0:
        raise Exception("The folder '%s' should be empty or not exist." % parameters.PREPROCESSED_DATA_FOLDER)
    set_total_docs(len(docs))
    dump_documents(docs)

def dump_documents(docs, path=None):
    from os.path import getsize
    os.makedirs(parameters.PREPROCESSED_DATA_FOLDER, exist_ok=True)
    size = 100000000 # 100 MB in bytes
    if path is not None:
        os.unlink(path)
    new_path = _generate_file_path()
    output_file = open(new_path, 'wb')
    for doc in docs:
        pickle.dump(obj=doc, file=output_file, protocol=2)
        if getsize(new_path) >= size:
            output_file.close()
            new_path = _generate_file_path()
            output_file = open(new_path, 'wb')
    output_file.close()
    return new_path

def check_data():
    for path in files_paths():
        docs = load_documents(path)
        for doc in docs:
            assert type(doc) is Document

def files_paths():
    names = os.listdir(parameters.PREPROCESSED_DATA_FOLDER)
    files_paths = []
    for name in names:
        files_paths.append(os.path.join(parameters.PREPROCESSED_DATA_FOLDER, name))
    files_paths.remove(_metadata_file)
    return files_paths

def _generate_file_path():
    from uuid import uuid4
    def generate_file_name():
        return ''.join([str(uuid4()), ".pkl"])
    names = files_paths()
    name = generate_file_name()
    while name in names:
        name = generate_file_name()
    return os.path.join(parameters.PREPROCESSED_DATA_FOLDER, name)

def get_total_docs():
    return load(_metadata_file)

def set_total_docs(n):
    dump(obj=n, path=_metadata_file)
