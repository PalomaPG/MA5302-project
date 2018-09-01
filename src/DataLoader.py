from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


class DataLoader(object):

    def __init__(self, path_file):
        self.path_file = path_file
        df_iterator = pd.read_csv(self.path_file, sep=',', index_col=['ATO_ID', 'objid'],
                                  header=0, engine='c', iterator=True, chunksize=100000)

        self.DF = pd.concat(df_iterator)
        self.DF = self.DF.drop(['ra', 'dec', 'fp_domper_o',  'fp_domper_c',
                                'fp_domperiod', 'starID', 'prob_CBF', 'prob_CBH',
                                'prob_DBF', 'prob_DBH', 'prob_HARD', 'prob_MIRA',
                                'prob_MPULSE', 'prob_MSINE', 'prob_NSINE', 'prob_PULSE',
                                'prob_SINE', 'prob_IRR', 'prob_LPV', 'prob_dubious'], axis=1)

    def label_prop(self):
        df = self.DF.loc[self.DF['CLASS'] != 'dubious']
        X = np.array(df.iloc[:, df.columns != 'CLASS'].values, dtype=float)
        y = np.array(df.iloc[:, df.columns == 'CLASS'].values, dtype=object)
        #y.reshape(y.shape[0], )
        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(X, y)
        return X, y

    def clustering(self):
        df = self.DF.loc[self.DF['CLASS'] == 'dubious']
        X = np.array(df.iloc[:, df.columns != 'CLASS'].values, dtype=float)
        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(X)
        return X