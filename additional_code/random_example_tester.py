#!/usr/bin/python3
# coding=utf-8

from getpass import getpass
from json import dumps
from pandas import read_excel, DataFrame
from random import randrange
from requests import post
from requests.auth import HTTPBasicAuth
from signal import signal, SIGINT
from sys import argv
from time import sleep

_continue = True

def initial_code_to_run_on_data_frame(data_frame):
    return data_frame

def main():
    global _continue
    if len(argv) != 10:
        print("Usage: python3 random_example_tester.py <Excel file> <text column> <classification column> <classifier> <url> <interval> <username> <accepted probabilities> <output file>")
        quit()
    excel_file = argv[1]
    text_column = argv[2]
    classification_column = argv[3]
    classifier = argv[4]
    URL = argv[5]
    interval = int(argv[6])
    username = argv[7]
    accepted_probs = int(argv[8])
    output_file = argv[9]
    password = getpass()
    print("Loading Excel file.")
    data_frame = read_excel(excel_file)
    data_frame = initial_code_to_run_on_data_frame(data_frame)
    print("Sending requests with random elements.")
    i = 0
    num_indexes = len(data_frame)
    out_df = DataFrame(columns=["Text", "Classification", "Prediction", "Valid prediction", "Accepted probabilities"])
    valid_predictions = 0
    signal(SIGINT, signal_handler)
    print('Press Ctrl+C to stop.')
    print("Iteration <number>: <valid predictions> / <invalid predictions>")
    while _continue:
        i = i + 1
        index = randrange(num_indexes)
        text = data_frame.loc[index, text_column]
        classification = data_frame.loc[index, classification_column]
        d = {
            'text': text,
            'classifier': classifier
        }
        response = post(url=URL, json=d, auth=HTTPBasicAuth(username, password))
        prediction = response.json()
        valid_prediction = classification in prediction[:accepted_probs]
        out_df.loc[i] = [text, classification, prediction, valid_prediction, accepted_probs]
        if valid_prediction:
            valid_predictions = valid_predictions + 1
        invalid_predictions = i - valid_predictions
        print("Iteration %s: %s / %s" % (i, valid_predictions, invalid_predictions))
        sleep(interval)
    print('Saving to Excel file...')
    out_df.to_excel("%s.xlsx" % (output_file))

def signal_handler(sig, frame):
    global _continue
    _continue = False

if __name__ == '__main__':
    main()
