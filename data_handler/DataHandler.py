import pygrib as grib
import numpy as np
import matplotlib.pyplot as plt

class DataHandler(object):

    def __init__(self, path_data):
        self.path = path_data

    def read_file(self):
        grbs = grib.open(self.path)
        for gr in grbs:
            print(gr)

