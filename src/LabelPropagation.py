from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

from sklearn.semi_supervised import LabelPropagation
from .Utils import plot_confusion_matrix


class MyLabelPropagation(object):

    def __init__(self, X, y):
        self.n_components = [10, 15, 25, 30]
        self.X = X
        self.y = y
        self.labels = np.unique(y)

    def preprocess(self, n_components, print_freq = False):
        X = preprocess_pca(self.X, self.y, n_components)
        if print_freq:
            unique, counts = np.unique(self.y, return_counts=True)
            print_dict(dict(zip(unique, counts)))
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.8, random_state=7)
        if print_freq:
            unique, counts = np.unique(y_train, return_counts=True)
            print_dict(dict(zip(unique, counts)))
            unique, counts = np.unique(y_test, return_counts=True)
            print_dict(dict(zip(unique, counts)))
        return X_train, y_train, X_test, y_test

    def process(self, n_components):
        X_train, y_train, X_test, y_test = self.preprocess(n_components)
        label_prop_model = LabelPropagation(n_jobs=-1)
        label_prop_model.fit(X_train, y_train)
        y_pred = label_prop_model.predict(X_test)
        mean_acc=label_prop_model.score(X_test, y_test)
        plot_confusion_matrix(y_test, y_pred, self.labels, normalize=False,
                              figname=('lp_comps_%d.png' % n_components))
        print(mean_acc)
        print(label_prop_model.get_params())

    def test(self):
        for i in self.n_components:
            self.process(i)

def print_dict(dict):
    print('.............')
    for k, v in dict.items():
        print(k+': '+str(v))
    print('...........\n')


def preprocess_pca(X, y, n_components):
    pca = PCA(n_components=n_components, random_state=7)
    X = pca.fit_transform(X, y)
    print(pca.get_params())
    print(pca.components_)
    return X