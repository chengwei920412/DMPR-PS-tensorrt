import logging, sys
from math import log10
from os import path
# sys.path.append("/home/hugoliu/github/DMPR-PS/trt/scripts/utils")

_NAME = "DMPR_PS"
_LEVEL = logging.DEBUG #logging.INFO

def init(log_dir, name):
    logger = logging.getLogger(_NAME)
    logger.setLevel(_LEVEL)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(levelname).1s%(asctime)s.%(msecs)03d    %(process)d %(filename)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(_LEVEL)
    logger.addHandler(console_handler)

    # Setup file logging
    file_handler = logging.FileHandler(path.join(log_dir, name + ".log"), mode="w")
    file_formatter = logging.Formatter(fmt="%(levelname).1s%(asctime)s.%(msecs)03d    %(process)d %(filename)s:%(lineno)d] %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(_LEVEL)
    logger.addHandler(file_handler)


def get_logger():
    return logging.getLogger(_NAME)
