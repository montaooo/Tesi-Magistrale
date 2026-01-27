import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime, time
from math import floor, isnan

from strategies.adversarial import poison_features
from tesseract import temporal, spatial

def best_K_data(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, botnet: str):
    '''
    Funzione per ottenere i K dati più significativi del dataset testato. Vengono scelti i K più vicini al decision boundary.
    '''
    
    if botnet == "all":
        probs = clf.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        
        idx_neg = np.where(y_test == 0)[0]
        idx_pos = np.where(y_test == 1)[0]
        k_neg = y_test[idx_neg].shape[0] // 2
        k_pos = y_test[idx_pos].shape[0] // 2

        neg_indexes = np.argsort(max_probs[idx_neg])[:k_neg]
        pos_indexes = np.argsort(max_probs[idx_pos])[:k_pos]
        indexes = np.concatenate([neg_indexes, pos_indexes])
    elif botnet == "single":
        K = y_test.shape[0] // 2
        indexes = np.argsort(max_probs)[:K]

    return X_test[indexes], y_test[indexes]

def decimal_floor(n: float, decimals: int):

    if n is None or isnan(n):
        return n
    n *= (10**decimals)
    n = floor(n)
    n /= (10**decimals)
    return n

def calculate_metrics(y_test, pred, results: dict[str, list], botnet="all"):
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel().tolist()

    if tp + fn == 0:
        tpr = "/"
    else:
        tpr = decimal_floor(tp / (tp + fn), 3)

    if tn + fp == 0:
        tnr = "/"
    else:
        tnr = decimal_floor(tn / (tn + fp), 3)

    if tp + fp + fn == 0 or botnet == "single":
        f1 = "/"
    else:
        f1 = decimal_floor((2*tp) / (2*tp + fp + fn), 3)
    
    if tp + fp == 0:
        precision = "/"
    else:
        precision = decimal_floor(tp / (tp + fp), 3)

    results["F1"].append(f1)
    results["Precision"].append(precision)
    results["TPR"].append(tpr)
    results["TNR"].append(tnr)
    
def check_importances(model, X_columns):
    importances = model.feature_importances_.tolist()
    imp_d = {}

    for c, v in zip(X_columns, importances):
        imp_d[c] = v

    imp_d = sorted(imp_d.items(), key=lambda x:-x[1])
    return imp_d

def check_importances_ensemble(ensemble_models, X_columns):
    all_data = []
    for clf in ensemble_models:
        imp_d = {}
        importances = clf.feature_importances_.tolist()
        for c, v in zip(X_columns, importances):
            imp_d[c] = v
        imp_d = sorted(imp_d.items(), key=lambda x:-x[1])
        all_data.append(imp_d)
    
    flattened_data = [item for sublist in all_data for item in sublist]
    df = pd.DataFrame(flattened_data, columns=['Feature', 'Importance'])
    mean_importances = df.groupby('Feature')['Importance'].mean().reset_index()
    mean_importances = mean_importances.sort_values(by='Importance', ascending=False)
    print(mean_importances.to_string(index=False))

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

def calculate_weights(y_test: pd.DataFrame, all_probs, botnet: str):
    weights = []
    for prob in all_probs:
        pred = np.argmax(prob, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel().tolist()
        if tp + fn == 0:
            tpr = 1
        else:
            tpr = tp / (tp + fn)

        if botnet == "all":
            if tn + fp == 0:
                tnr = 1
            else:
                tnr = tn / (tn + fp)
            weights.append(decimal_floor(np.sqrt(tpr * tnr), 3))
        else:
            weights.append(decimal_floor(tpr, 3))
    
    return weights

def update_buffer(buffer_X, buffer_y, new_X, new_y, size: int):
    if buffer_X is None:
        return new_X, new_y
    
    X_combined = np.vstack((buffer_X, new_X))
    y_combined = np.concatenate((buffer_y, new_y))

    idx_neg = np.where(y_combined == 0)[0]
    idx_pos = np.where(y_combined == 1)[0]
    
    half = size // 2

    selected_neg = np.random.choice(idx_neg, min(len(idx_neg), half), replace=False)
    selected_pos = np.random.choice(idx_pos, min(len(idx_pos), half), replace=False)

    indexes = np.concatenate([selected_neg, selected_pos])
    
    return X_combined[indexes], y_combined[indexes]

def most_important_features(max_probs, X_test: pd.DataFrame, y_test: pd.DataFrame, K=10000):
    '''
    Funzione per ottenere le feature più importanti dato un dataset e le probabilità di decisione ottenute dal modello. Le migliori probabilità non sono altro che quelle più vicine al 50% di decisione tra una classe e l'altra (perché significa che sono le feature più vicine al decision boundary, cioè quelle più "importanti per il modello").
    
    :param max_probs: Indice delle migliori probabilità
    :param X_test 
    :param y_test
    :param K: Valore che indica quanti dati prendere per ogni label. Di default è a 10000, quindi ci saranno 10000 dati benevoli e 10000 dati malevoli.
    '''
    idx_neg_full = np.where(y_test == 0)[0]
    idx_pos_full = np.where(y_test == 1)[0]
    
    neg_indexes = idx_neg_full[np.argsort(max_probs[idx_neg_full])[:K]]
    pos_indexes = idx_pos_full[np.argsort(max_probs[idx_pos_full])[:K]]

    X_neg = X_test[neg_indexes]
    y_neg = y_test[neg_indexes]
    X_pos = X_test[pos_indexes]
    y_pos = y_test[pos_indexes]

    return X_neg, y_neg, X_pos, y_pos

def cl_mu(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    fixed_start_date = pd.to_datetime("2016-09-01")
    train_size = 8
    ensemble_models = []
    weights = []
    target_samples = 50000
    X_columns = dset.drop(columns=['Date', 'Label']).columns

    # Divisione temporale dei dataset
    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    X_tests, y_tests, t_tests = clean_dsets(X_tests, y_tests, t_tests)

    # Sampling 1:1
    X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(25, 25), batch_size=10, alpha=0.1, max_iter=250, random_state=42)
    # clf = LogisticRegression(C=0.01, class_weight={0: 5, 1: 1}, random_state=42)
    # scaler = StandardScaler()
    # -------------------------------- TRAINING --------------------------------
    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    # X_train_past = scaler.fit_transform(X_train_past)
    # X_test_past = scaler.transform(X_test_past)
    clf.fit(X_train_past, y_train_past)
    ensemble_models.append(clf)
    weights.append(1)
    pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, X_test_past, weights)
    t_train = pd.Series(t_train)
    results['Date'].append(f"{t_train.max().month}-{t_train.max().year}")
    calculate_metrics(y_test_past, pred, results, botnet)

    max_probs = np.max(avg_probs, axis=1)
    buffer_X_neg, buffer_y_neg, buffer_X_pos, buffer_y_pos = most_important_features(max_probs, X_test_past, y_test_past, K=target_samples)
    
    # -------------------------------- TESTING --------------------------------
    print("Start Testing")
    starttime = time.time()
    print(f"Start time: {datetime.datetime.now().time()}")

    # Adversarial data
    features_to_increment = ["Dur", "SrcBytes", "TotPkts"]
    increments = [10, 16, 10]

    for i, (X_test, y_test, t_test) in enumerate(zip(X_tests, y_tests, t_tests), 1):
        t_test: pd.Series
        print(f"Cycle {i}")
        # X_test = scaler.transform(X_test)   
        unlearning = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}

        # Adversarial function
        poison_features(X_test, y_test, X_columns, features_to_increment, increments)


        pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, X_test, weights)
        max_probs = np.max(avg_probs, axis=1)

        # -------------------------------- CONTINUAL LEARNING --------------------------------

        # Controllo e aggiusto la quantità di dati benevoli e malevoli per il retraining        
        X_test_neg, y_test_neg, X_test_pos, y_test_pos = most_important_features(max_probs, X_test, y_test, K=target_samples)
        
        if len(y_test_neg) < target_samples:
            diff = target_samples - len(y_test_neg)
            X_test_neg = np.vstack([X_test_neg, buffer_X_neg[:diff]])
            y_test_neg = np.concatenate([y_test_neg, buffer_y_neg[:diff]])
        if len(y_test_pos) < target_samples:
            diff = target_samples - len(y_test_pos)
            X_test_pos = np.vstack([X_test_pos, buffer_X_pos[:diff]])
            y_test_pos = np.concatenate([y_test_pos, buffer_y_pos[:diff]])
        
        # Aggiornamento buffer
        buffer_X_neg = X_test_neg
        buffer_y_neg = y_test_neg
        buffer_X_pos = X_test_pos
        buffer_y_pos = y_test_pos

        X_new_ensemble = np.vstack([X_test_neg, X_test_pos])
        y_new_ensemble = np.concatenate([y_test_neg, y_test_pos])


        weights = calculate_weights(y_test, all_probs, botnet)

        # -------------------------------- MACHINE UNLEARNING --------------------------------

        if len(ensemble_models) > 5:
            if i != 1:
                total_tpr = results['TPR'][-1]
                total_tnr = results['TNR'][-1]
                total_f1 = results['F1'][-1]

                total_tpr = 0 if type(total_tpr) == str else total_tpr
                total_tnr = 0 if type(total_tnr) == str else total_tnr
                total_f1 = 0 if type(total_f1) == str else total_f1

                # Calcolo il rendimento dell'ensemble senza un modello (a turno)
                for j in range(len(ensemble_models)):
                    models_left = ensemble_models[:j] + ensemble_models[j+1:]
                    weights_lef = weights[:j] + weights[j+1:]
                    pred, all_probs, avg_probs = ensemble_predict_weighted(models_left, X_test, weights_lef)
                    calculate_metrics(y_test, pred, unlearning, botnet)
                
                # Calcolo il modello col rendimento peggiore e valuto se eliminarlo
                bonus = 0.01
                unlearning = {k: [1 if x == '/' else x for x in v] for k, v in unlearning.items()}
                unlearning['F1'] = [el + bonus for el in unlearning['F1']]
                unlearning['TNR'] = [el + bonus for el in unlearning['TNR']]
                unlearning['TPR'] = [el + bonus for el in unlearning['TPR']]

                max_index = unlearning['F1'].index(max(unlearning['F1']))
                
                if unlearning['F1'][max_index] >= total_f1 and unlearning['TNR'][max_index] >= total_tnr and unlearning['TPR'][max_index] >= total_tpr:
                    del ensemble_models[max_index]
                    del weights[max_index]
                    print(f"Rimozione modello {max_index}")

        weights.append(1)

        results['Date'].append(f"{t_test.iloc[0].month}-{t_test.iloc[0].year}")
        calculate_metrics(y_test, pred, results, botnet)

        # print(len(np.where(y_new_ensemble == 0)[0]), len(np.where(y_new_ensemble == 1)[0]))
        ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        # ensemble_models.append(MLPClassifier(hidden_layer_sizes=(25, 25), batch_size=10, max_iter=250, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        # ensemble_models.append(LogisticRegression(C=0.01, class_weight={0: 5, 1: 1}, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        print(f"Numero modelli: {len(ensemble_models)}")

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}")
    print(results)
    # check_importances_ensemble(ensemble_models, X_columns)
    return results

def cl(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    fixed_start_date = pd.to_datetime("2016-09-01")
    train_size = 8
    ensemble_models = []
    weights = []
    target_samples = 50000
    X_columns = dset.drop(columns=['Date', 'Label']).columns

    # Divisione temporale dei dataset
    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    X_tests, y_tests, t_tests = clean_dsets(X_tests, y_tests, t_tests)

    # Sampling 1:1
    X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(25, 25), batch_size=10, alpha=0.1, max_iter=250, random_state=42)
    # clf = LogisticRegression(C=0.01, class_weight={0: 5, 1: 1}, random_state=42)
    # scaler = StandardScaler()
    # -------------------------------- TRAINING --------------------------------
    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)

    # X_train_past = scaler.fit_transform(X_train_past)
    # X_test_past = scaler.transform(X_test_past)
    clf.fit(X_train_past, y_train_past)
    ensemble_models.append(clf)
    weights.append(1)
    pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, X_test_past, weights)
    t_train = pd.Series(t_train)
    results['Date'].append(f"{t_train.max().month}-{t_train.max().year}")
    calculate_metrics(y_test_past, pred, results, botnet)

    # Creazione buffer dei migliori dati recenti 
    max_probs = np.max(avg_probs, axis=1)
    buffer_X_neg, buffer_y_neg, buffer_X_pos, buffer_y_pos = most_important_features(max_probs, X_test_past, y_test_past, K=target_samples)
    
    # -------------------------------- TESTING --------------------------------
    print("Start Testing")
    starttime = time.time()
    print(f"Start time: {datetime.datetime.now().time()}")

    # Adversarial data
    features_to_increment = ["Dur", "SrcBytes", "TotPkts"]
    increments = [10, 16, 10]

    for i, (X_test, y_test, t_test) in enumerate(zip(X_tests, y_tests, t_tests), 1):
        t_test: pd.Series
        print(f"Cycle {i}")
        # X_test = scaler.transform(X_test)
        # Adversarial function
    
        poison_features(X_test, y_test, X_columns, features_to_increment, increments)

        pred, all_probs, avg_probs = ensemble_predict_weighted(ensemble_models, X_test, weights)
        max_probs = np.max(avg_probs, axis=1)

        # -------------------------------- CONTINUAL LEARNING --------------------------------

        # Controllo e aggiusto la quantità di dati benevoli e malevoli per il retraining        
        X_test_neg, y_test_neg, X_test_pos, y_test_pos = most_important_features(max_probs, X_test, y_test, K=target_samples)
        
        if len(y_test_neg) < target_samples:
            diff = target_samples - len(y_test_neg)
            X_test_neg = np.vstack([X_test_neg, buffer_X_neg[:diff]])
            y_test_neg = np.concatenate([y_test_neg, buffer_y_neg[:diff]])
        if len(y_test_pos) < target_samples:
            diff = target_samples - len(y_test_pos)
            X_test_pos = np.vstack([X_test_pos, buffer_X_pos[:diff]])
            y_test_pos = np.concatenate([y_test_pos, buffer_y_pos[:diff]])
        
        # Aggiornamento buffer dei dati completi più recenti
        buffer_X_neg = X_test_neg
        buffer_y_neg = y_test_neg
        buffer_X_pos = X_test_pos
        buffer_y_pos = y_test_pos

        X_new_ensemble = np.vstack([X_test_neg, X_test_pos])
        y_new_ensemble = np.concatenate([y_test_neg, y_test_pos])

        weights = calculate_weights(y_test, all_probs, botnet)
        weights.append(1)

        results['Date'].append(f"{t_test.iloc[0].month}-{t_test.iloc[0].year}")
        calculate_metrics(y_test, pred, results, botnet)

        # print(len(np.where(y_new_ensemble == 0)[0]), len(np.where(y_new_ensemble == 1)[0]))
        ensemble_models.append(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        # ensemble_models.append(MLPClassifier(hidden_layer_sizes=(25, 25), batch_size=10, max_iter=250, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        # ensemble_models.append(LogisticRegression(C=0.01, class_weight={0: 5, 1: 1}, random_state=42).fit(X_new_ensemble, y_new_ensemble))
        print(f"Numero modelli: {len(ensemble_models)}")

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}")
    print(results)
    # check_importances_ensemble(ensemble_models, X_columns)
    return results

def concept_drift(dset: pd.DataFrame, test_size, botnet: str):
    results = {"Precision": [], "F1": [], "TNR": [], "TPR": [], "Date": []}
    t = pd.to_datetime(dset['Date'])
    y = dset['Label'].values
    X = dset.drop(columns=['Date', 'Label']).values
    fixed_start_date = pd.to_datetime("2016-09-01")
    train_size = 8
    X_columns = dset.drop(columns=['Date', 'Label']).columns

    # Divisione temporale dei dataset
    splits = temporal.time_aware_train_test_split(X, y, t, train_size=train_size, test_size=1, granularity="month", start_date=fixed_start_date)
    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
    X_tests, y_tests, t_tests = clean_dsets(X_tests, y_tests, t_tests)

    # Sampling 1:1
    X_train, y_train, t_train = spatial.downsample_set(X_train, y_train, t_train.values, min_pos_rate=1/2)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(25,25), batch_size=10, max_iter=250, alpha=0.1, random_state=42)
    # scaler = StandardScaler()
    # clf = LogisticRegression(C=0.01, class_weight={0: 5, 1: 1}, random_state=42)
    # -------------------------------- TRAINING --------------------------------
    print("Old data training...")
    X_train_past, X_test_past, y_train_past, y_test_past = train_test_split(X_train, y_train, test_size=test_size, random_state=42, stratify=y_train)
    # X_train_past = scaler.fit_transform(X_train_past)
    # X_test_past = scaler.transform(X_test_past)
    
    clf.fit(X_train_past, y_train_past)
    
    pred = clf.predict(X_test_past)
    # scores = clf.decision_function(X_test_past)
    # threshold = 0.1
    # pred = (scores > threshold).astype(int)
    t_train = pd.Series(t_train)
    results['Date'].append(f"{t_train.max().month}-{t_train.max().year}")
    calculate_metrics(y_test_past, pred, results, botnet)

    # -------------------------------- TESTING --------------------------------
    print("Testing...")

    starttime = time.time()
    print(f"Start time: {datetime.datetime.now().time()}")

    # Adversarial data "Dur", "SrcBytes", "TotPkts"
    features_to_increment = ["Dur", "SrcBytes", "TotPkts"]
    increments = [10, 16, 10]
    
    for i, (X_test, y_test, t_test) in enumerate(zip(X_tests, y_tests, t_tests), 1):
        # X_test = scaler.transform(X_test)
        t_test: pd.Series
        print(f"Cycle {i}")
        results['Date'].append(f"{t_test.iloc[0].month}-{t_test.iloc[0].year}")
        
        # Adversarial function
        
        poison_features(X_test, y_test, X_columns, features_to_increment, increments)

        pred = clf.predict(X_test)
        calculate_metrics(y_test, pred, results, botnet)

    endtime = time.time() - starttime
    print(f"Time taken: {endtime}")
    print(results)
    # print(check_importances(clf, X_columns))
    
    return results
