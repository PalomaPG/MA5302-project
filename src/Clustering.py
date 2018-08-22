import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


class Clustering(object):

    def __init__(self, path_file, n_clusters=14):
        self.n_clusters = 14
        self.path_file = path_file
        '''
        self.DF_train = None
        self.DF_train_labels = None
        self.DF_test = None
        self.DF_test_labels = None
        '''
        self.DF = None
        self.DF_labels = None

    def load_data(self):
        df_iterator = pd.read_csv(self.path_file, sep=',', index_col=['ATO_ID', 'objid'],header=0, engine='c', iterator=True, chunksize=100000)
        DF = pd.concat(df_iterator)
        DF = DF.drop(['ra', 'dec', 'fp_domper_o', 'fp_domperiod', 'starID'], axis=1)
        DF = DF.loc[DF['CLASS'] != 'dubious']
        self.DF = np.array(DF.iloc[:, DF.columns != 'CLASS'].values, dtype=float)
        self.DF_labels = (np.array(DF.iloc[:, DF.columns == 'CLASS'].values, dtype=object))
        self.DF_labels = self.DF_labels.reshape(self.DF_labels.shape[0], )
        #unique, counts = np.unique(self.DF_labels, return_counts=True)
        #print(dict(zip(unique, counts)))
        #print(DF.shape)
        #plot_histo(self.DF_labels, 'without_dubious.png')
        '''
        n_rows = DF.shape[0]
        self.DF_train = DF.iloc[: int(n_rows*.75), :]
        self.DF_train_labels = np.array(self.DF_train['CLASS'])
        self.DF_test = DF.iloc[ int(n_rows*.75):, :]
        self.DF_test_labels = np.array(self.DF_test['CLASS'])
        self.DF_train = np.array(self.DF_train.drop(['CLASS'], axis=1).astype(float))
        self.DF_test = np.array(self.DF_test.drop(['CLASS'], axis=1).astype(float))
        #print(self.DF_train.isna().sum())
        #print(self.DF_test.isna().sum())
        '''
    def preprocess(self, n_components):
        DF = preprocess_pca(self.DF, self.DF_labels, n_components)
        return DF

    def kmeans_clust(self, X):
        kmeans = cluster.KMeans(n_clusters=self.n_clusters,
                                random_state=7)
        predicted = kmeans.fit_predict(X)
        return count_class_assigment(predicted, self.DF_labels)

    def mini_kmeans_clust(self, X, batch_size=100):
        mini_kmeans = cluster.MiniBatchKMeans(n_clusters=self.n_clusters,
                                              batch_size=batch_size, random_state=7)
        predicted = mini_kmeans.fit_predict(X)
        return count_class_assigment(predicted, self.DF_labels)

    def dbscan_clust(self, X, eps=1.0, min_samples=5, metric='euclidean'):
        dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        predicted = dbscan.fit_predict(X)
        return count_class_assigment(predicted, self.DF_labels)

    def plot_misclassify_count(self):
        n_feats = [2, 5, 10, 25, 50, 75, 100, 125]
        means = []
        stds = []
        for n in n_feats:
            X = self.preprocess(n)
            #dbscan_cls = self.dbscan_clust(X)
            kmeans_cls = self.kmeans_clust(X)
            mini_kmeans_cls = self.mini_kmeans_clust(X)
            cls_lst = [kmeans_cls, mini_kmeans_cls]
            mean, std = clust_statistics(cls_lst)
            means.append(mean)
            stds.append(std)
        means = np.array(means).T
        stds = np.array(stds).T
        plt.clf()
        for i in range(means.shape[0]):
            plt.errorbar(n_feats, means[i], yerr=stds[i], fmt='o')
        plt.title('# Assigments')
        plt.xlabel('# features')
        plt.savefig('test.png')


def clust_statistics(clust_lst):
    class_assig_mean = []
    class_assig_std = []
    for dictio in clust_lst:
        print(dictio)
        len_reg = []
        for k, v in dictio.items():
            len_reg.append(len(v))
        class_assig_mean.append(np.mean(np.array(len_reg)))
        class_assig_std.append(np.std(np.array(len_reg)))
    return class_assig_mean, class_assig_std

def count_class_assigment(predicted, actual_labels):
    diff_class = {}
    labels_lst = np.unique(actual_labels).tolist()
    for l in labels_lst:
        diff_class[l] = []
    labels = actual_labels.tolist()
    for i in range(0, len(labels)):
        if predicted[i] not in diff_class[labels[i]]:
            diff_class[labels[i]].append(predicted[i])
    return diff_class

def plot_histo(y,filename):
    plt.clf()
    print('Plotting...')
    plt.hist(y)
    plt.title('Frecuencia de clases')
    plt.xlabel('Tipo de estrella variable')
    plt.ylabel('Frecuencia')
    plt.savefig(filename)
    #plt.show()


def preprocess_pca(X, y, n_components):
    scaler = StandardScaler(copy=False)
    X=scaler.fit_transform(X, y)
    pca = PCA(n_components=n_components, random_state=7)
    X = pca.fit_transform(X, y)
    return X