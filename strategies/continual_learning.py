import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import datetime, time
from math import floor

from tesseract import temporal, spatial

def best_K_data(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, K: int):
    '''
    Funzione per ottenere i K dati più significativi del dataset testato. Vengono scelti i K più vicini al decision boundary.
    '''
    # K = y_test.shape[0] // 2
    probs = clf.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    indexes = np.argsort(max_probs)[:K]

    return X_test[indexes], y_test[indexes]

def decimal_floor(n: float, decimals: int):
    n *= (10**decimals)
    n = floor(n)
    n /= (10**decimals)
    return n

def calculate_metrics(y_test, pred, results, botnet="all"):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel().tolist()
    tpr = decimal_floor(tp / (tp + fn), 3)
    if botnet == 'all':
        tnr = decimal_floor(tn / (tn + fp), 3)
        f1 = decimal_floor((2*tp) / (2*tp + fp + fn), 3)
        precision = decimal_floor(tp / (tp + fp), 3)

        results["F1"].append(f1)
        results["Precision"].append(precision)
        results["TPR"].append(tpr)
        results["TNR"].append(tnr)
    elif botnet == "single":
        results["TPR"].append(tpr)
        results["TNR"].append("/")
        results["Precision"].append("/")
        results["F1"].append("/")
    else:
        raise ValueError(f"name '{botnet}' invalid")

def check_importances(model, X_columns):
    importances = model.feature_importances_.tolist()
    imp_d = {}

    for c, v in zip(X_columns, importances):
        imp_d[c] = v

    imp_d = sorted(imp_d.items(), key=lambda x:-x[1])
    return imp_d

def clean_dsets(X_tests, y_tests, t_tests):
    X_tests_puliti = []
    y_tests_puliti = []
    t_tests_puliti = []
    for i, x_test in enumerate(X_tests):
        if x_test.shape[0] > 0:
            X_tests_puliti.append(x_test)
            y_tests_puliti.append(y_tests[i])
            t_tests_puliti.append(t_tests[i])
    
    return X_tests_puliti, y_tests_puliti, t_tests_puliti

def calculate_dates(start_date: pd.Timestamp, train_size: int, results, t_tests):
    '''
    Inserimento in results dei vari set di train e test.
    '''
    cur_year = start_date.year
    cur_month = (start_date.month + train_size) % 12
    if cur_month == 0:
        cur_month = 12
    if start_date.day == 1:
        cur_month -= 1
    cur_year += (start_date.month + train_size) // 12

    results["Date"].append(f"{cur_month}/{cur_year}")
    for t in t_tests:
        cur_month = t.iloc[0].month
        cur_year = t.iloc[0].year

        if start_date.day != 1:
            if cur_month == 12:
                results["Date"].append(f"{start_date.day}/{cur_month}/{cur_year} - {start_date.day - 1}/{1}/{cur_year+1}")
            else:    
                results["Date"].append(f"{start_date.day}/{cur_month}/{cur_year} - {start_date.day - 1}/{cur_month + 1}/{cur_year}")
        else:
            results["Date"].append(f"{cur_month}/{cur_year}")

def print_metrics(metrics, f):
    m = pd.DataFrame(metrics)
    m.set_index('Date', inplace=True)

    print(m.to_markdown(numalign="center", stralign="center"), file=f)
    print("", file=f)
    
def splits_handle(splits, botnet):
    '''
    Nel caso di tutte le botnet:
    Per avere dataset con più dati e il maggior numero di casi di test, divido in Maggio, Giugno/Luglio, Agosto/Settembre, Aprile/Maggio 2018.
    Per farlo, i test saranno: 8 Maggio - 7 Giugno, 8 Giugno - 7 Agosto, 8 Agosto - 7 Settembre, 8 Aprile - 7 Maggio 2018 (unisco i mesi Giugno e Luglio perché avrei pochi dati normali altrimenti).
    
    :param splits: Array contenente i vari dataset di train e test periodici iniziali
    '''

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    
    # Rimozione mesi in cui non ci sono dati
    X_tests, y_tests, t_tests = clean_dsets(X_tests, y_tests, t_tests)

    if botnet == "all":
        X_optimized_tests = []
        y_optimized_tests = []
        t_optimized_tests = []

        i = 0
        while i < len(X_tests):
            if i == 1:
                X_optimized_tests.append(np.concatenate([X_tests[i], X_tests[i+1]]))
                y_optimized_tests.append(np.concatenate([y_tests[i], y_tests[i+1]]))
                t_optimized_tests.append(pd.concat([t_tests[i], t_tests[i+1]]))
                i += 2
            else:
                X_optimized_tests.append(X_tests[i])
                y_optimized_tests.append(y_tests[i])
                t_optimized_tests.append(t_tests[i])
                i += 1
    elif botnet == "single":
        X_optimized_tests = X_tests
        y_optimized_tests = y_tests
        t_optimized_tests = t_tests
    else:
        raise ValueError(f"botnet name '{botnet}' invalid")      
    
    return X_train, y_train, t_train, X_optimized_tests, y_optimized_tests, t_optimized_tests

#def ensemble_predict(ensemble, X: pd.DataFrame):
    '''
    Funzione per effettuare la predizione dell'ensemble. La votazione avviene a maggioranza di confidenza.
    
    :ensemble: Lista di tutti i modelli.
    :X: Set di dati da testare.
    '''
    if not ensemble:
        raise ValueError("Empty ensemble")

    all_probs = [model.predict_proba(X) for model in ensemble]
    avg_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    return final_preds, all_probs, avg_probs

def ensemble_predict_weighted(ensemble, X: pd.DataFrame, weights):
    if not ensemble:
        raise ValueError("Empty ensemble")
    
    all_probs = [model.predict_proba(X) for model in ensemble]
    weights = np.array(weights)
    weigthed_probs = np.array(all_probs) * weights[:, np.newaxis, np.newaxis]
    sum_probs = np.sum(weigthed_probs, axis=0)
    sum_weights = np.sum(weights)
    final_probs = sum_probs / sum_weights
    final_preds = np.argmax(final_probs, axis=1)

    return final_preds, all_probs, final_probs

def calculate_weights(y_test: pd.DataFrame, all_probs):
    weights = []
    for prob in all_probs:
        pred = np.argmax(prob, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel().tolist()
        tpr = round(tp / (tp + fn), 3)
        weights.append(tpr)
    
    return weights

def rf_cl(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    X_columns = dset.drop(columns=['Date', 'Label']).columns
    fixed_start_date = pd.to_datetime("2016-09-08")
    train_size = 8

    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    
    X_train, y_train, t_train, X_optimized_tests, y_optimized_tests, t_optimized_tests = splits_handle(splits, botnet)
    calculate_dates(fixed_start_date, train_size, results, t_optimized_tests)

    # Downsample dati di training (per performance) e testing (troppi malware)
    if botnet == "all":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)

        for i, (x_i, y_i, t_i) in enumerate(zip(X_optimized_tests, y_optimized_tests, t_optimized_tests)):
            x_i, y_i, t_i = spatial.downsample_set(x_i, y_i, t_i.values, min_pos_rate=1/21)
            X_optimized_tests[i] = x_i
            y_optimized_tests[i] = y_i
            t_optimized_tests[i] = t_i
    elif botnet == "single":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/21)
    else:
        raise ValueError(f"botnet name '{botnet}' invalid")

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    clf.fit(X_train_past, y_train_past)
    pred = clf.predict(X_test_past)
    calculate_metrics(y_test_past, pred, results, botnet)

    K = y_train.shape[0] // 2
    X_train_sliding, y_train_sliding = best_K_data(clf, X_train, y_train, K)

    # -------------------- CONTINUAL LEARNING --------------------    
    starttime = time.time()
    f = open("performances/tmp.txt", "a")
    print(f"Start time: {datetime.datetime.now().time()}", file=f)

    for i, (x_test, y_test) in enumerate(zip(X_optimized_tests, y_optimized_tests), 1):
        print(f"Cycle {i}")
        
        clf.fit(X_train_sliding, y_train_sliding)
        pred = clf.predict(x_test)
        # print(check_importances(clf, X_columns))
        calculate_metrics(y_test, pred, results, botnet)
        
        K = y_test.shape[0] // 2
        X_retraining, y_retraining = best_K_data(clf, x_test, y_test, K)
        X_train_sliding = np.vstack((X_train_sliding, X_retraining))
        y_train_sliding = np.concatenate((y_train_sliding, y_retraining))

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}", file=f)

    print_metrics(results, f)
    f.close()
    return results

def ensemble_cl(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    X_columns = dset.drop(columns=['Date', 'Label']).columns
    fixed_start_date = pd.to_datetime("2016-09-01")
    train_size = 8
    ensemble_models = []

    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    
    X_train, y_train, t_train, X_optimized_tests, y_optimized_tests, t_optimized_tests = splits_handle(splits, botnet)
    calculate_dates(fixed_start_date, train_size, results, t_optimized_tests)

    # Downsample dati di training (per performance)
    if botnet == "all":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    elif botnet == "single":
        X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/21)
    else:
        raise ValueError(f"botnet name '{botnet}' invalid")

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    clf.fit(X_train_past, y_train_past)
    ensemble_models.append(clf)
    pred = clf.predict(X_test_past)
    calculate_metrics(y_test_past, pred, results, botnet)

    mask_neg = (y_train == 0)
    K = np.sum(y_train == 0) // 2
    X_neg, y_neg = best_K_data(clf, X_train[mask_neg], y_train[mask_neg], K)
    
    # -------------------- CONTINUAL LEARNING --------------------

    starttime = time.time()
    f = open("performances/tmp.txt", "a")
    print(f"Start time: {datetime.datetime.now().time()}", file=f)

    weights = []
    for i, (x_test, y_test) in enumerate(zip(X_optimized_tests, y_optimized_tests), 1):
        print(f"Cycle {i}")
        weights.append(1)
        pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, x_test, weights)
        # print(check_importances(clf, X_columns))
        calculate_metrics(y_test, pred, results, botnet)
        
        max_probs = np.max(avg_probs, axis=1)
        K = y_test.shape[0] // 2
        indexes = np.argsort(max_probs)[:K]

        weights = calculate_weights(y_test, all_probs)

        if botnet == "all":
            ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(x_test[indexes], y_test[indexes]))
        elif botnet == "single":
            X_sliding = np.vstack((X_neg, x_test[indexes]))
            y_sliding = np.concatenate((y_neg, y_test[indexes]))
            ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_sliding, y_sliding))

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}", file=f)

    print_metrics(results, f)
    f.close()
    return results