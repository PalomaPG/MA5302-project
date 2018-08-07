import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

class Clustering(object):

    def __init__(self, path_file, n_clusters=13):
        self.path_file = path_file
        self.clustering_method = None
        self.clustering_method = cluster.KMeans(n_clusters=n_clusters, algorithm='auto',
                                                random_state=7)
        self.DF_train = None
        self.DF_train_labels = None
        self.DF_test = None
        self.DF_test_labels = None

    def load_data(self):
        df_iterator = pd.read_csv(self.path_file, sep=',', index_col=['ATO_ID', 'objid'],header=0, engine='c', iterator=True, chunksize=100000)
        DF = pd.concat(df_iterator)
        DF = DF.drop(['ra', 'dec', 'fp_domper_o', 'fp_domperiod', 'starID'], axis=1)
        n_rows = DF.shape[0]

        self.DF_train = DF.iloc[: int(n_rows*.75), :]
        self.DF_train_labels = np.array(self.DF_train['CLASS'])
        self.DF_test = DF.iloc[ int(n_rows*.75):, :]
        self.DF_test_labels = np.array(self.DF_test['CLASS'])
        self.DF_train = np.array(self.DF_train.drop(['CLASS'], axis=1).astype(float))
        self.DF_test = np.array(self.DF_test.drop(['CLASS'], axis=1).astype(float))
        #print(self.DF_train.isna().sum())
        #print(self.DF_test.isna().sum())

    def clusterization(self):
        self.clustering_method.fit(self.DF_train)
        for i in range(len(self.DF_train)):
            pred = np.array(self.DF_train[i])
            pred = pred.reshape(-1, len(pred))
            prediction = self.clustering_method.predict(pred)
            print(prediction)