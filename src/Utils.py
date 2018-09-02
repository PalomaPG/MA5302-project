import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def plot_confusion_matrix(y, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figname='test.png'):

    cm = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figname)


def simple_plot(x, ys, title, xlabel, ylabel, labels, figname='test.png'):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cycol = itertools.cycle('bgrcmk')
    for i in range(len(ys)):
        plt.plot(x, ys[i], '-o', alpha=.6, mfc=next(cycol), label=labels[i])
    plt.legend()
    plt.savefig(figname)


def preprocess_pca(X, y, n_components):
    pca = PCA(n_components=n_components, random_state=7)
    cols = X.columns
    print(cols[104], cols[109], cols[101], cols[4], cols[5])

    X = pca.fit_transform(X, y)
    print(cols[np.abs(pca.components_[0]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[1]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[2]).argsort()[::-1][:3]])
    """
    print(cols[np.abs(pca.components_[3]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[4]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[5]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[6]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[7]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[8]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[9]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[10]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[11]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[12]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[13]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[14]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[15]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[16]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[17]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[18]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[19]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[20]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[21]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[22]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[23]).argsort()[::-1][:3]])
    print(cols[np.abs(pca.components_[24]).argsort()[::-1][:3]])
    """
    #print(pd.DataFrame(pca.components_, columns=columns))
    return X


def print_dict(dict):
    print('.............')
    for k, v in dict.items():
        print(str(k)+': '+str(v))
    print('...........\n')


def create_dict(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


def tsne_vis(X, y_pred, metric='canberra'):
    l, y = np.unique(y_pred, return_inverse=True)
    y = y.tolist()
    X = TSNE(n_components=3, init='pca', metric=metric).fit_transform(X)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1],  X[:, 2], c=y)
    plt.savefig('tsne_canberra3.png')
