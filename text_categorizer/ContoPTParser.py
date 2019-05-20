#!/usr/bin/python3
# coding=utf-8

from re import compile

class ContoPTParser:
    pattern1 = compile(r'\d+ : \w+ : (?:.+\(\d+\.\d+\);)+')
    pattern2 = compile(r'\s?(.+)\(\d+\.\d+\)')
    
    def __init__(self, filename):
        self.synonyms = self._load_synonyms(filename)

    def _load_synonyms(self, filename):
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
