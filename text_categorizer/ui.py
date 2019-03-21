#!/usr/bin/python3
# coding=utf-8

def verify_python_version():
    from functions import get_python_version
    from logger import logger
    version_array = get_python_version()
    if version_array < [3,5]:
        logger.error("Please use Python3.5 or higher.")
        quit()
