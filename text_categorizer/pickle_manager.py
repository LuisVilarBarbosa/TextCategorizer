#!/usr/bin/python3
# coding=utf-8

import pickle
import os
import re
import parameters

from tqdm import tqdm
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

def dump_document(doc):
    if not os.path.isdir(parameters.PREPROCESSED_DATA_FOLDER):
        os.makedirs(parameters.PREPROCESSED_DATA_FOLDER)
    path = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, ''.join(["doc-", str(doc.index), ".pkl"]))
    dump(obj=doc, path=path)

def load_all_documents():
    folder_path = parameters.PREPROCESSED_DATA_FOLDER
    names = os.listdir(folder_path)
    pattern = re.compile(r'.*-(\d+).*') # TODO: fix (There is no match if the name does not contains "-".)
    names.sort(key=lambda n: int(pattern.match(n).group(1)))
    docs = []
    for name in tqdm(iterable=names, desc="Loading documents", unit="doc"):
        path = os.path.join(parameters.PREPROCESSED_DATA_FOLDER, name)
        doc = load(path)
        assert type(doc) is Document
        docs.append(doc)
    return docs

def dump_all_documents(docs):
    for doc in tqdm(iterable=docs, desc="Storing documents", unit="doc"):
        dump_document(doc=doc)
