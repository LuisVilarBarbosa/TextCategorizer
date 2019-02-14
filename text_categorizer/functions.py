#!/usr/bin/python3
# coding=utf-8

def get_python_version():
    from sys import version
    version_number = int(version[:version.find(".")])
    return version_number
