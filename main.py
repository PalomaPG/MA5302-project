import sys
from src.DataLoader import DataLoader
from src.LabelPropagation import MyLabelPropagation

def main(path_):
    dl = DataLoader(path_)
    dl.clustering()
    #lp = MyLabelPropagation(X, y)
    #lp.test()

if __name__ == '__main__':
    main(sys.argv[1])