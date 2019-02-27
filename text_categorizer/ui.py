#!/usr/bin/python3
# coding=utf-8

def verify_python_version():
    from functions import get_python_version
    version_number = int(get_python_version())
    if version_number < 3:
        print("Please use Python3 or higher.")
        quit()
