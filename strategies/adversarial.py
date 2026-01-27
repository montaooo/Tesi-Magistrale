import pandas as pd
import numpy as np

def poison_dset(X, y: pd.DataFrame, X_columns: pd.Index, ratio=0.4):
    X_p = X.copy()
    botnet_indexes = np.where(y == 1)[0]
    benign_indexes = np.where(y == 0)[0]

    poison_indexes = np.random.choice(botnet_indexes, int(len(botnet_indexes) * ratio), replace=False)
    min_pkt_sz = X_columns.get_loc('dMinPktSz')
    max_pkt_sz = X_columns.get_loc('dMaxPktSz')

    if len(benign_indexes) > 0:
        mean_benign = np.mean(X_p[benign_indexes, min_pkt_sz])
        mean_benign = np.random.uniform(mean_benign-10, mean_benign+10, len(poison_indexes))
    else:
        mean_benign = np.random.uniform(90,110, len(poison_indexes))
    
    X_p[poison_indexes, min_pkt_sz] = mean_benign
    
    # Devo gestire i casi inconsistenti, se la dimensione massima è minore della minima le rendo uguali.
    X_p[poison_indexes, max_pkt_sz] = np.maximum(X_p[poison_indexes, max_pkt_sz], X_p[poison_indexes, min_pkt_sz])
    return X_p

def dttl_dtos_poison(X, y: pd.DataFrame, X_columns: pd.Index, ratio=0.4):
    X_p = X.copy()
    botnet_indexes = np.where(y == 1)[0]
    benign_indexes = np.where(y == 0)[0]
    poison_indexes = np.random.choice(botnet_indexes, int(len(botnet_indexes) * ratio), replace=False)

    dttl = X_columns.get_loc('dTtl')
    dtos = X_columns.get_loc('dTos')

    if len(benign_indexes) > 0:
        mean_dttl = np.mean(X_p[benign_indexes, dttl])
        mean_dtos = np.mean(X_p[benign_indexes, dtos])
        mean_dttl = np.random.uniform(mean_dttl-10, mean_dttl+10, len(poison_indexes))
        mean_dtos = np.random.uniform(mean_dtos-10, mean_dtos+10, len(poison_indexes))
    else:
        mean_dttl = np.random.uniform(70,100, len(poison_indexes))
        mean_dtos = 0.0

    X_p[poison_indexes, dttl] = mean_dttl
    X_p[poison_indexes, dtos] = mean_dtos    

    return X_p

def poison_features(X, y, X_columns: pd.Index, features: list, increments: list):
    botnet_indexes = np.where(y == 1)[0]
    feature_columns = [X_columns.get_loc(column) for column in features]
    botnet_indexes = np.random.choice(botnet_indexes, int(len(botnet_indexes) * 0.3), replace=False)
    
    idx = {name: X_columns.get_loc(name) for name in ["Dur", "SrcDur", "SrcBytes", "TotBytes", "TotPkts", "SrcPkts", "Rate", "SrcRate", "Load", "SrcLoad", "sMaxPktSz", "dMaxPktSz"]}
    for c_index, i, f in zip(feature_columns, increments, features):
        X[botnet_indexes, c_index] += i

        if f == "Dur":
            X[botnet_indexes, idx['SrcDur']] += i
        elif f ==  "SrcBytes":
            X[botnet_indexes, idx['TotBytes']] += i
        elif f == "TotPkts":
            X[botnet_indexes, idx['SrcPkts']] += i
        
    X[botnet_indexes, idx['Rate']] = (X[botnet_indexes, idx['TotPkts']]-1)/X[botnet_indexes, idx['Dur']]
    X[botnet_indexes, idx['SrcRate']] = (X[botnet_indexes, idx['SrcPkts']]-1)/X[botnet_indexes, idx['SrcDur']]
    X[botnet_indexes, idx['Load']] = ((X[botnet_indexes, idx['TotBytes']] - (X[botnet_indexes, idx['sMaxPktSz']]+X[botnet_indexes, idx['dMaxPktSz']]))*8)/X[botnet_indexes, idx['Dur']]
    X[botnet_indexes, idx['SrcLoad']] = ((X[botnet_indexes, idx['SrcBytes']] - X[botnet_indexes, idx['sMaxPktSz']])*8)/X[botnet_indexes, idx['SrcDur']]