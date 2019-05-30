import logging
from os.path import abspath
from pathlib import Path

filename="log.txt"
logformat="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
dateformat="%Y-%m-%d %H:%M:%S %z %Z"

path = abspath(__file__)
package = Path(path).parent.name
logger = logging.getLogger(package)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt=logformat, datefmt=dateformat)

file_handler = logging.FileHandler(filename, 'a', 'utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
