# Developed on January 29 and 30, 2020.

import os
import time
from collections import OrderedDict
from subprocess import check_call
from sys import argv

def get_most_important_features(get_most_important_features_py, clfs):
    for clf in clfs:
        check_call([
            'python', get_most_important_features_py, 'vectorizer.pkl', '{clf}.pkl'.format(clf=clf),
            '100000000000000000', '{clf} features.json'.format(clf=clf), '{clf} features.xlsx'.format(clf=clf)
        ])

def generate_new_filenames(desc, machine, clfs):
    files_times = OrderedDict()
    filename = 'log.txt'
    if os.path.exists(filename):
        files_times[filename] = os.path.getctime(filename)
    files_times['predictions.json'] = os.path.getmtime('predictions.json')
    for clf in clfs:
        for ext in ['.json', '.xlsx']:
            filename = '{clf}{ext}'.format(clf=clf, ext=ext)
            files_times[filename] = os.path.getmtime(filename)

    mapping = OrderedDict()
    for old_name, t in files_times.items():
        filename, ext = os.path.splitext(old_name)
        new_name = '{time} {desc} - {filename} ({machine}){ext}'.format(
            time=time.strftime('%Y-%m-%d %Hh%Mm%S', time.localtime(t)), desc=desc, filename=filename, machine=machine, ext=ext)
        mapping[old_name] = new_name
    return mapping

def show_mapping(mapping):
    for old_name, new_name in mapping.items():
        print('- {0}\n+ {1}\n'.format(old_name, new_name))

def rename_files(mapping):
    for old_name, new_name in mapping.items():
        os.rename(old_name, new_name)

def main():
    if len(argv) >= 5:
        get_most_important_features_py = os.path.abspath(argv[1])
        path = argv[2]
        desc = argv[3]
        machine = argv[4]
        clfs = argv[5:]
        old_dir = os.getcwd()
        os.chdir(path)
        get_most_important_features(get_most_important_features_py, clfs)
        mapping = generate_new_filenames(desc, machine, clfs)
        show_mapping(mapping)
        if input('Rename files (y/n)? ').lower() == 'y':
            rename_files(mapping)
            print('Renaming completed.')
        else:
            print('Renaming cancelled.')
        os.chdir(old_dir)
    else:
        print('Usage: %s <get_most_important_features_py_path> <path> <description> <machine_name> [<classifier>]*' % (argv[0]))

if __name__ == '__main__':
    main()
