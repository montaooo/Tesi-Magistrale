import pandas as pd
import numpy as np

def poison_features(X, y, X_columns: pd.Index, features: list, increments: list):
    botnet_indexes = np.where(y == 1)[0]
    feature_columns = [X_columns.get_loc(column) for column in features]
    botnet_indexes = np.random.choice(botnet_indexes, int(len(botnet_indexes) * 1), replace=False)
    idx = {name: X_columns.get_loc(name) for name in ["Dur", "SrcDur", "SrcBytes", "TotBytes", "TotPkts", "SrcPkts", "Rate", "SrcRate", "Load", "SrcLoad", "sMaxPktSz", "dMaxPktSz"]}
    
    for c_index, i, f in zip(feature_columns, increments, features):
        X[botnet_indexes, c_index] += i

        if f == "Dur":
            X[botnet_indexes, idx['SrcDur']] += i
        elif f ==  "SrcBytes":
            X[botnet_indexes, idx['TotBytes']] += i
        elif f == "TotPkts":
            X[botnet_indexes, idx['SrcPkts']] += i
    
    
    X[botnet_indexes, idx['Rate']] = np.divide(X[botnet_indexes, idx['TotPkts']]-1, X[botnet_indexes, idx['Dur']], out=X[botnet_indexes, idx['Rate']], where= X[botnet_indexes, idx['Dur']] != 0)
    X[botnet_indexes, idx['SrcRate']] = np.divide(X[botnet_indexes, idx['SrcPkts']]-1, X[botnet_indexes, idx['SrcDur']], out=X[botnet_indexes, idx['SrcRate']], where=X[botnet_indexes, idx['SrcDur']] != 0)
    X[botnet_indexes, idx['Load']] = np.divide((X[botnet_indexes, idx['TotBytes']] - (X[botnet_indexes, idx['sMaxPktSz']]+X[botnet_indexes, idx['dMaxPktSz']]))*8, X[botnet_indexes, idx['Dur']], out=X[botnet_indexes, idx['Load']], where=X[botnet_indexes, idx['Dur']] != 0)
    X[botnet_indexes, idx['SrcLoad']] = np.divide((X[botnet_indexes, idx['SrcBytes']] - X[botnet_indexes, idx['sMaxPktSz']])*8, X[botnet_indexes, idx['SrcDur']], out=X[botnet_indexes, idx['SrcLoad']], where=X[botnet_indexes, idx['SrcDur']] != 0)

    
    