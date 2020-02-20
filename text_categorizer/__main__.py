import os
import sys
from text_categorizer import prediction_server, trainer
from text_categorizer.Parameters import Parameters
from text_categorizer.ui import verify_python_version

def main(argv):
    verify_python_version()
    n_args = len(argv)
    mode = argv[1] if 1 < n_args else None
    if mode == "--trainer" and n_args == 3:
        config_file = argv[2]
        parameters = Parameters(config_file)
        os.makedirs(parameters.data_dir, exist_ok=True)
        old_dir = os.getcwd()
        os.chdir(parameters.data_dir)
        trainer.main(parameters)
        os.chdir(old_dir)
    elif mode == "--prediction_server" and n_args == 4:
        config_file = argv[2]
        port = int(argv[3])
        parameters = Parameters(config_file)
        os.makedirs(parameters.data_dir, exist_ok=True)
        old_dir = os.getcwd()
        os.chdir(parameters.data_dir)
        prediction_server.main(parameters, port)
        os.chdir(old_dir)
    else:
        print("Usage: python3 -m text_categorizer --trainer <configuration file>")
        print("       python3 -m text_categorizer --prediction_server <configuration file> <port>")

if __name__ == "__main__":
    main(sys.argv)
