#!/usr/bin/python3
# coding=utf-8

import pickle
import os
from text_categorizer.Document import Document
from uuid import uuid4

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

def get_documents(filename):
    input_file = open(filename, 'rb')
    metadata = pickle.load(input_file)
    total = metadata['total']
    for _ in range(total):
        yield pickle.load(input_file)
    try:
        pickle.load(input_file)
        raise "The file '%s' has more documents than indicated in the metadata." % (input_file)
    except EOFError:
        input_file.close()

def dump_documents(docs, filename):
    if os.path.exists(filename):
        raise Exception("The file '%s' should not exist." % filename)
    metadata = {'total': len(docs)}
    pda = PickleDumpAppend(metadata=metadata, filename=filename)
    for doc in docs:
        pda.dump_append(doc)
    pda.close()

def check_data(filename):
    total = 0
    docs = get_documents(filename)
    for doc in docs:
        assert type(doc) is Document
        total = total + 1
        assert doc.index + 1 == total
    metadata = get_docs_metadata(filename)
    assert metadata['total'] == total

def _generate_file():
    while True:
        filename = ''.join([str(uuid4()), ".pkl"])
        if not os.path.exists(filename):
            open(file=filename, mode='w').close()
            return filename

def get_docs_metadata(filename):
    input_file = open(filename, 'rb')
    metadata = pickle.load(input_file)
    input_file.close()
    return metadata

class PickleDumpAppend():
    def __init__(self, metadata, filename):
        assert type(metadata) is dict
        assert type(filename) is str
        self.filename_upon_completion = filename
        my_filename = _generate_file()
        self.file = open(file=my_filename, mode='wb')
        pickle.dump(obj=metadata, file=self.file, protocol=_pickle_protocol)
    
    def dump_append(self, data):
        pickle.dump(obj=data, file=self.file, protocol=_pickle_protocol)
    
    def close(self):
        self.file.close()
        if os.path.exists(self.filename_upon_completion):
            os.remove(self.filename_upon_completion)
        os.rename(self.file.name, self.filename_upon_completion)
