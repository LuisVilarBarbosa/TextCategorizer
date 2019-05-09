#!/usr/bin/python3
# coding=utf-8

import pandas
import re
import sys

def count_occurrences(filename, classification_column):
    def update_count(code_to_match, counts):
        if code_to_match in counts:
            counts[code_to_match] = counts[code_to_match] + 1
        else:
            counts[code_to_match] = 1
    classifications = pandas.read_excel(filename)[classification_column]
    pattern = r'(?:\')(\w+(?:\.\w+)*)(?: \-)'
    counts = dict()
    for c in classifications:
        results = re.findall(pattern, c)
        for result in results:
            code = result
            code_parts = code.split(".")
            code_to_match = code_parts[0]
            update_count(code_to_match, counts)
            for i in range(1, len(code_parts)):
                code_to_match = code_to_match + "." + code_parts[i]
                update_count(code_to_match, counts)
    return counts

def print_counts(counts):
    for k,v in sorted(counts.items()):
        print("%s: %s" % (k, v))

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 count_occurrences.py <excel file> <classification column>")
        quit()
    filename = sys.argv[1]
    classification_column = sys.argv[2]
    counts = count_occurrences(filename, classification_column)
    print_counts(counts)

if __name__ == '__main__':
    main()
