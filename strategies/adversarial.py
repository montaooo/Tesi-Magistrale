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