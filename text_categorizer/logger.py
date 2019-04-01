import logging

filename="log.txt"
logformat="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
dateformat="%Y-%m-%d %H:%M:%S %z %Z"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt=logformat, datefmt=dateformat)

file_handler = logging.FileHandler(filename, 'a', 'utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
