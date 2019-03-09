#!/usr/bin/python3
# coding=utf-8

import json
import jsonpickle
import os
import parameters

from Document import Document

def dump(obj, path):
    output_file = open(path, 'w')
    json_string = jsonpickle.dumps(obj)
    output_file.write(json_string)
    output_file.close()

def load(path):
    input_file = open(path, 'r')
    json_string = input_file.read()
    input_file.close()
    data = jsonpickle.loads(json_string)
    return data

def dump_document(doc, index):
    if not os.path.isdir(parameters.PREPROCESSED_DATA_FOLDER):
        os.makedirs(parameters.PREPROCESSED_DATA_FOLDER)
    path = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, ''.join(["doc-", str(index), ".json"]))
    dump(obj=doc, path=path)

def load_all_documents():
    folder_path = parameters.PREPROCESSED_DATA_FOLDER
    names = os.listdir(folder_path)
    names = sorted(names)
    docs = []
    for name in names:
        path = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, name)
        try:
            doc = load(path)
            assert type(doc) is Document
            docs.append(doc)
        except json.decoder.JSONDecodeError:
            print("An error occurred decoding the file: %s" % path)
            exit(0)
    return docs

def dump_all_documents(docs):
    i = 0
    for doc in docs:
        i = i + 1
        dump_document(doc=doc, index=i)
