#!/usr/bin/python3
# coding=utf-8

import pickle
import os
import parameters

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

def get_documents():
    input_file = open(parameters.PREPROCESSED_DATA_FILE, 'rb')
    _total = pickle.load(input_file)
    while True:
        try:
            yield pickle.load(input_file)
        except EOFError:
            input_file.close()
            break

def dump_documents(docs):
    if os.path.exists(parameters.PREPROCESSED_DATA_FILE):
        raise Exception("The file '%s' should not exist." % parameters.PREPROCESSED_DATA_FILE)
    pda = PickleDumpAppend(total=len(docs), filename=parameters.PREPROCESSED_DATA_FILE)
    for doc in docs:
        pda.dump_append(doc)
    pda.close()

def check_data():
    from Document import Document
    total = 0
    docs = get_documents()
    for doc in docs:
        assert type(doc) is Document
        total = total + 1
        assert doc.index - 1 == total
    assert get_total_docs() == total

def _generate_file():
    def generate_filename():
        from uuid import uuid4
        return ''.join([str(uuid4()), ".pkl"])
    filename = generate_filename()
    while os.path.exists(filename):
        filename = generate_filename()
    open(file=filename, mode='w').close()
    return filename

def get_total_docs():
    input_file = open(parameters.PREPROCESSED_DATA_FILE, 'rb')
    total = pickle.load(input_file)
    input_file.close()
    return total

class PickleDumpAppend():
    def __init__(self, total, filename):
        assert type(total) is int
        assert type(filename) is str
        self.filename_upon_completion = filename
        my_filename = _generate_file()
        self.file = open(file=my_filename, mode='wb')
        pickle.dump(obj=total, file=self.file, protocol=_pickle_protocol)
    
    def dump_append(self, data):
        pickle.dump(obj=data, file=self.file, protocol=_pickle_protocol)
    
    def close(self):
        self.file.close()
        if os.path.exists(self.filename_upon_completion):
            os.remove(self.filename_upon_completion)
        os.rename(self.file.name, self.filename_upon_completion)
