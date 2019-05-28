#!/usr/bin/python3
# coding=utf-8

from sys import argv
from prediction_server import main as prediction_server_main
from trainer import main as trainer_main
from ui import verify_python_version

def main():
    verify_python_version()
    n_args = len(argv)
    mode = argv[1] if 1 < len(argv) else None
    if not ((n_args == 3 and mode == "--trainer") or (n_args == 4 and mode == "--prediction_server")):
        print("Usage: python3 text_categorizer --trainer <configuration file>")
        print("       python3 text_categorizer --prediction_server <configuration file> <port>")
        quit()
    if mode == "--trainer":
        config_file = argv[2]
        trainer_main(config_file)
    elif mode == "--prediction_server":
        config_file = argv[2]
        port = int(argv[3])
        prediction_server_main(config_file, port)
    else:
        raise ValueError("Only the options '--trainer' and '--prediction_server' are accepted.")

if __name__ == "__main__":
    main()
