from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.semi_supervised import LabelPropagation
from .Utils import plot_confusion_matrix, simple_plot, preprocess_pca, print_dict


class MyLabelPropagation(object):

    def __init__(self, X, y):
        self.n_components = [25]
        print(type(X))
        self.X = X
        self.y = y
        self.labels = np.unique(y)
        self.m_acc = []

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
        self.m_acc.append(mean_acc)
        print(label_prop_model.get_params())

    def test(self):
        for i in self.n_components:
            self.process(i)
        print(self.m_acc)
        simple_plot(self.n_components, [self.m_acc], title='Norm. accuracy per n components',
                    ylabel='Accuracy', xlabel='N of components', labels=['Norm. Acc'],
                    figname='acc_lp.png')




