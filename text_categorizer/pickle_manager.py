#!/usr/bin/python3
# coding=utf-8

import pickle
import os
import parameters

_metadata_file = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, "metadata.pkl")
_pickle_protocol = 2

def dump(obj, path):
    output_file = open(path, 'wb')
    pickle.dump(obj, output_file, _pickle_protocol)
    output_file.close()

def load(path):
    input_file = open(path, 'rb')
    data = pickle.load(input_file)
    input_file.close()
    return data

def load_all_documents():
    from numpy import append
    filenames = filenames()
    docs = []
    for filename in filenames:
        ds = load_documents(filename)
        docs = append(docs, ds)
    docs = docs.flatten().tolist()
    docs.sort(key=lambda doc: doc.index)
    return docs

def load_documents(filename):
    path = _get_file_path(filename)
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
    filename = _generate_file()
    update_files_on_metadata(filename)
    dump_documents(docs, filename)

def dump_documents(docs, filename):
    os.makedirs(parameters.PREPROCESSED_DATA_FOLDER, exist_ok=True)
    pda = PickleDumpAppend(filename)
    for doc in docs:
        pda.dump_append(doc)
    pda.close()

def check_data():
    from Document import Document
    metadata = load(_metadata_file)
    total = 0
    for filename in filenames():
        docs = load_documents(filename)
        for doc in docs:
            assert type(doc) is Document
            total = total + 1
            assert doc.index - 1 == total
    assert metadata["total"] == total

def filenames():
    filenames = []
    try:
        data = load(_metadata_file)
        if "files" in data:
            filenames = data["files"]
    except FileNotFoundError:
        pass
    return filenames

def _get_file_path(filename):
    return os.path.join(parameters.PREPROCESSED_DATA_FOLDER, filename)

def _generate_file():
    def generate_filename():
        from uuid import uuid4
        return ''.join([str(uuid4()), ".pkl"])
    path = _get_file_path("")
    filenames = os.listdir(path)
    filename = generate_filename()
    while filename in filenames:
        filename = generate_filename()
    path = _get_file_path(filename)
    open(file=path, mode='w').close()
    return filename

def update_files_on_metadata(new_filename, last_filename=None):
    from numpy import append, where, insert, array
    try:
        data = load(_metadata_file)
    except FileNotFoundError:
        data = dict()
    if "files" in data:
        if last_filename is None:
            data["files"] = append(data["files"], new_filename)
        else:
            indexes = where(data["files"] == last_filename)
            data["files"] = insert(data["files"], indexes[0] + 1, new_filename)
    else:
        data["files"] = array(new_filename)
    data["files"] = data["files"].flatten()
    dump(data, _metadata_file)


def get_total_docs():
    data = load(_metadata_file)
    return data["total"]

def set_total_docs(n):
    try:
        data = load(_metadata_file)
    except FileNotFoundError:
        data = dict()
    data["total"] = n
    dump(obj=data, path=_metadata_file)

class PickleDumpAppend():
    def __init__(self, filename):
        from numpy import array
        path = _get_file_path(filename)
        self.filename = filename
        self.file = open(file=path, mode='wb')
    
    def dump_append(self, data):
        from os.path import getsize
        size = 100000000 # 100 MB in bytes
        pickle.dump(obj=data, file=self.file, protocol=_pickle_protocol)
        if getsize(self.file.name) >= size:
            self.file.close()
            filename = _generate_file()
            update_files_on_metadata(new_filename=filename, last_filename=self.filename)
            self.filename = filename
            path = _get_file_path(filename)
            self.file = open(file=path, mode='wb')
    
    def close(self):
        self.file.close()

