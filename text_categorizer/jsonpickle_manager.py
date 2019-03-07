#!/usr/bin/python3
# coding=utf-8

import jsonpickle

def dump(obj, filename):
    output_file = open(filename, 'w')
    json_string = jsonpickle.dumps(obj)
    output_file.write(json_string)
    output_file.close()

def load(filename):
    input_file = open(filename, 'r')
    json_string = input_file.read()
    input_file.close()
    data = jsonpickle.loads(json_string)
    return data
