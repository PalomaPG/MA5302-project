import sys
from src.DataLoader import DataLoader
from src.Clustering import Clustering
from src.LabelPropagation import MyLabelPropagation

def main(path_):
    dl = DataLoader(path_)
    X = dl.clustering()
    clust = Clustering(X)
    clust.process()
    #X, y= dl.label_prop()
    #lp = MyLabelPropagation(X, y)
    #lp.test()

if __name__ == '__main__':
    main(sys.argv[1])