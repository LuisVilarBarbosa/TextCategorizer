from getpass import getpass
from json import dumps
from pandas import read_excel
from random import randrange
from requests import post
from requests.auth import HTTPBasicAuth
from sys import argv
from time import sleep

def initial_code_to_run_on_data_frame(data_frame):
    return data_frame

def main():
    if len(argv) != 8:
        print("Usage: python3 random_example_tester.py <Excel file> <text column> <classification column> <classifier> <url> <interval> <username>")
        quit()
    excel_file = argv[1]
    text_column = argv[2]
    classification_column = argv[3]
    classifier = argv[4]
    URL = argv[5]
    interval = int(argv[6])
    username = argv[7]
    password = getpass()
    print("Loading Excel file.")
    data_frame = read_excel(excel_file)
    data_frame = initial_code_to_run_on_data_frame(data_frame)
    print("Sending requests with random elements.")
    i = 0
    num_indexes = len(data_frame)
    while True:
        i = i + 1
        index = randrange(num_indexes)
        text = data_frame.loc[index, text_column]
        classification = data_frame.loc[index, classification_column]
        d = {
            'text': text,
            'classifier': classifier
        }
        response = post(url=URL, json=d, auth=HTTPBasicAuth(username, password))
        data = response.json()
        prediction = data[0]
        print("Iteration %s: Classification=%s Prediction=%s" % (i, classification, prediction))
        if classification != prediction:
            print("\tText:\n%s" % (text))
        sleep(interval)

if __name__ == '__main__':
    main()
