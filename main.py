import sys, os
from src.Clustering import Clustering


def main(path_):
    clust = Clustering(path_)
    clust.load_data()
    clust.clusterization()


if __name__ == '__main__':
    main(sys.argv[1])