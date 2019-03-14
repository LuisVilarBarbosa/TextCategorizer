import logging

filename="log.txt"
logformat="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
dateformat="%Y-%m-%d %H:%M:%S %z %Z"
logging.basicConfig(
    filename=filename,
    level=logging.DEBUG,
    format=logformat,
    datefmt=dateformat)
logger = logging.getLogger(filename)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(logformat)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
