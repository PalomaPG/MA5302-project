import pygrib as grib
import itertools
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
from os import listdir
from os import path
import os


class DataHandler(object):

    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.index_xrange = (240, 261)
        self.index_yrange = (140, 161)
        self.aux_lats = None
        self.aux_lons = None
        self.pressure_threshold = 1000


    def write_csv(self):
        """
        Escribe CSV desde datos grb2
        :return:
        """
        for root, dirs, files in os.walk(self.path_dir):
            for filename in files:
                date = root.split('/')[-1]
                hour = filename.split('.')[1][1:3]
                print("date: %s, time: %s, archivo: %s" % (date, hour, filename))
                self.read_file(root, filename)



    def read_file(self, root, filename):
        """
        Lee archivo GRIB2
        :param root: directorio donde se encuentra filename
        :param filename: nombre del archivo grib2
        :return:
        """
        grbs = grib.open(root+'/'+filename)
        for g in grbs:
            print(g)
        grbs = grbs.select(name='Pressure')
        for g in grbs:
            print(g)

            if self.aux_lons is None and self.aux_lats is None:
                #grb[(g.typeOfLevel, g.level)]= g.values
                lats, lons = g.latlons()
                self.aux_lats =lats[240:261, 550:600] # Index lat: 30-40 South
                self.aux_lons =lons[240:261, 550:600]
                print(self.aux_lons)
                print(np.min(self.aux_lats[:, 0]), np.max(self.aux_lats[:, 0]))
                print(np.min(self.aux_lons[0, :]), np.max(self.aux_lons[0, :]))

            median = np.median(g.values[240:261, 140:161])
            ax = sns.heatmap(g.values)
            #plt.show()

            #plt.imshow(g.values[240:261, 140:161], cmap='hot', interpolation='nearest')
            #plt.savefig('hola.png')
            #print(np.nanmin(np.nanmin(g.values[240:261, 140:161], axis=0)))

    def create_latlon_lst(self):
        for i in range(0, 21):
            for j in range(0, 21):
                for lat, lon in itertools.product(self.aux_lats[i], self.aux_lons[j]):
                    print('%f,%f' % (lat, lon))

