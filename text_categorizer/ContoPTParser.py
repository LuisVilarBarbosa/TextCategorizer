#!/usr/bin/python3
# coding=utf-8

import io
import re
import requests
from os.path import exists
from zipfile import ZipFile

class ContoPTParser:
    pattern1 = re.compile(r'\d+ : \w+ : (?:.+\(\d+\.\d+\);)+')
    pattern2 = re.compile(r'\s?(.+)\(\d+\.\d+\)')
    
    def __init__(self, filename):
        if not exists(filename):
            r = requests.get(url='http://ontopt.dei.uc.pt/recursos/CONTO.PT.01.zip', stream=True)
            with ZipFile(io.BytesIO(r.content)) as archive:
                data = archive.read('contopt_0.1_r2_c0.0.txt')
                f = open(filename, 'wb')
                f.write(data)
                f.close()
        self.synonyms = ContoPTParser._load_synonyms(filename)

    @staticmethod
    def _load_synonyms(filename):
        f = open(filename, "r", encoding="utf-8")
        d = dict()
        line = f.readline()
        while line != "":
            if ContoPTParser.pattern1.match(line):
                synonyms_and_conf = line.split(":")[2]
                synonyms_and_conf = synonyms_and_conf.split(";")
                synonyms_and_conf.pop()
                synonyms = [ContoPTParser.pattern2.match(synonym_and_conf).group(1) for synonym_and_conf in synonyms_and_conf]
                best = synonyms[0]
                for i in range(1, len(synonyms)):
                    d[synonyms[i]] = best
            line = f.readline()
        f.close()
        return d
