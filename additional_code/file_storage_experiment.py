#!/usr/bin/python3
# coding=utf-8

import jsonpickle
import pickle
import _pickle
import ujson
import json
import marshal

from pandas import read_excel
from Document import Document
from parameters import EXCEL_FILE

def file_storage_experiment():
    from sys import getsizeof
    from timeit import timeit

    n = 100000
    data_frame = read_excel(EXCEL_FILE)
    d = Document(data_frame, 1)

    t = timeit("jsonpickle.dumps([1,2,3])", "import jsonpickle", number=n)
    print("jsonpickle.dumps(): %ss" % t)

    t = timeit("jsonpickle.loads(d)", "import jsonpickle\nd=jsonpickle.dumps([1,2,3])", number=n)
    print("jsonpickle.loads(): %ss" % t)

    t = timeit("pickle.dumps([1,2,3])", "import pickle", number=n)
    print("pickle.dumps(): %ss" % t)

    t = timeit("pickle.loads(d)", "import pickle\nd=pickle.dumps([1,2,3])", number=n)
    print("pickle.loads(): %ss" % t)

    t = timeit("_pickle.dumps([1,2,3])", "import _pickle", number=n)
    print("_pickle.dumps(): %ss" % t)

    t = timeit("_pickle.loads(d)", "import _pickle\nd=_pickle.dumps([1,2,3])", number=n)
    print("_pickle.loads(): %ss" % t)

    t = timeit("ujson.dumps([1,2,3])", "import ujson", number=n)
    print("ujson.dumps(): %ss" % t)

    t = timeit("ujson.loads(d)", "import ujson\nd=ujson.dumps([1,2,3])", number=n)
    print("ujson.loads(): %ss" % t)

    t = timeit("json.dumps([1,2,3])", "import json", number=n)
    print("json.dumps(): %ss" % t)

    t = timeit("json.loads(d)", "import json\nd=json.dumps([1,2,3])", number=n)
    print("json.loads(): %ss" % t)

    t = timeit("marshal.dumps([1,2,3])", "import marshal", number=n)
    print("marshal.dumps(): %ss" % t)

    t = timeit("marshal.loads(d)", "import marshal\nd=marshal.dumps([1,2,3])", number=n)
    print("marshal.loads(): %ss" % t)

    print("Size of simple jsonpickle dump: %s bytes" % getsizeof(jsonpickle.dumps([1,2,3])))
    print("Size of simple pickle dump: %s bytes" % getsizeof(pickle.dumps([1,2,3])))
    print("Size of simple _pickle dump: %s bytes" % getsizeof(_pickle.dumps([1,2,3])))
    print("Size of simple ujson dump: %s bytes" % getsizeof(ujson.dumps([1,2,3])))
    print("Size of simple json dump: %s bytes" % getsizeof(json.dumps([1,2,3])))
    print("Size of simple marshal dump: %s bytes" % getsizeof(marshal.dumps([1,2,3])))

    print("Size of complex jsonpickle dump: %s bytes" % getsizeof(jsonpickle.dumps(d.__dict__)))
    print("Size of complex pickle dump: %s bytes" % getsizeof(pickle.dumps(d.__dict__)))
    print("Size of complex _pickle dump: %s bytes" % getsizeof(_pickle.dumps(d.__dict__)))
    try:
        print("Size of complex ujson dump: %s bytes" % getsizeof(ujson.dumps(d.__dict__)))
    except Exception as e:
        print("Size of complex ujson dump: Error - %s" % e)
    try:
        print("Size of complex json dump: %s bytes" % getsizeof(json.dumps(d.__dict__)))
    except Exception as e:
        print("Size of complex json dump: Error - %s" % e)
    print("Size of complex marshal dump: %s bytes" % getsizeof(marshal.dumps(d.__dict__)))

if __name__ == "__main__":
    file_storage_experiment()
