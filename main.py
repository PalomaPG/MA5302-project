import sys, os
from data_handler.DataHandler import DataHandler


def main(path):
    dh = DataHandler(path)
    dh.read_file()


if __name__ == '__main__':
    main(sys.argv[1])