import pandas as pd
import numpy as np
from  sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from .Utils import create_dict, preprocess_pca, print_dict, tsne_vis


class Clustering(object):

    def __init__(self, X):
        self.X = X

    def process(self):
        self.X = preprocess_pca(self.X, y=None, n_components=3)
        y_dbscan = self.dbscan_clust()
        print_dict(create_dict(y_dbscan))
        #tsne_vis(self.X[:2500], y_dbscan[:2500])

    def dbscan_clust(self, eps=0.5, min_samples=3, metric='canberra',
                     metric_params={}, algorithm='auto'):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                                algorithm=algorithm, metric_params=metric_params)
        y_dbscan = dbscan.fit_predict(self.X)
        return y_dbscan
    '''
    def tune_dbscan(self, n,  eps=0.5, min_samples=5, metric='euclidean', algorithm='auto',
                    filename='test.csv'):
        X = self.preprocess(n)
        dbscan_cls, predicted = self.dbscan_clust(X, eps=eps, min_samples=min_samples,
                                       metric=metric, algorithm=algorithm)

        labels = np.unique(self.DF_labels)
        save_table_results(labels, self.DF_labels, predicted, n=self.n_clusters, filename=filename)
        print(dbscan_cls)
    '''


