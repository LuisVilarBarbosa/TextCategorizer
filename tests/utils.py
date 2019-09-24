from itertools import zip_longest
from os import close, write, unlink
from os.path import exists
from tempfile import mkstemp
from uuid import uuid4

example_excel_file = 'example_excel_file.xlsx'
default_encoding = 'utf-8'
config_file = 'config.ini'

def decode(binary_string):
    s = binary_string.decode(default_encoding)
    s = s.replace('\r', '')
    return s

def create_temporary_file(content=None, text=False):
    fd, path = mkstemp(suffix=None, prefix=None, dir=None, text=text)
    if content is not None:
        write(fd, content)
    close(fd)
    return path

def remove_and_check(path):
    unlink(path)
    assert not exists(path)

def generate_available_filename():
	while True:
	    filename = str(uuid4())
	    if not exists(filename):
	        return filename
