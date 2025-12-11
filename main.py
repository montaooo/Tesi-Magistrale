import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re

from concept_drift import rf_downsampling, rf_oversampling, standard_rf, rf_periodic_drift, rf_monthly
from strategies.continual_learning import rf_cl, ensemble_cl, decimal_floor

pd.options.mode.chained_assignment = None

class FlowFile:
    def __init__(self, year, month, day, name, index, attack=None):
        self.year = year
        self.month = month
        self.day = day
        self.index = index
        self.attack = attack
        self.name = name
        self.date = "{}-{}-{}".format(self.year, self.month, self.day)
        self.create_filename()

    def create_filename(self):
        self.filename = f"{self.date}{self.name}{self.index}.csv"

def fixValues(dset: pd.DataFrame):
    # Funzione per rimuovere valori NaN e infiniti
    # NaN -> valore medio
    # Infiniti -> Valore massimo
    for c in dset.columns:
        if dset[c].dtype == 'int' or dset[c].dtype == 'float':
            tmp = np.asarray(dset[c], dtype=np.float64)
            tmp = tmp[np.isfinite(tmp)]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                mean_val = tmp.mean()

            if (str(mean_val) == 'nan'):
                mean_val = -1
                max_val = -1
            else:
                max_val = tmp.max()
    
            dset.replace({c: [np.inf, -np.inf]}, max_val, inplace=True)
            dset.replace({c: [np.nan]}, mean_val, inplace=True)
    dset = dset.dropna(subset=['Dir'])
    return dset

def fixPorts(dset: pd.DataFrame):
    dset['Sport_num'] = dset['Sport']
    dset['Dport_num'] = dset['Dport']
    # NaN -> -1
    dset.replace({'Sport_num': [np.nan]}, '-1', inplace=True)
    dset.replace({'Dport_num': [np.nan]}, '-1', inplace=True)
    # Hex -> -1
    dset["Sport_num"] = np.where(dset['Sport_num'].str.contains("0x"), '-1', dset['Sport_num'])
    dset["Dport_num"] = np.where(dset['Dport_num'].str.contains("0x"), '-1', dset['Dport_num'])
    
    dset['Sport_num'] = pd.to_numeric(dset['Sport_num'], errors='coerce')
    dset['Dport_num'] = pd.to_numeric(dset['Dport_num'], errors='coerce')
    # NaN -> 0 perché well-known (servizi standard: HTTP, FTP, SSH, ...)
    dset.replace({'Sport_num': [np.nan]}, 0, inplace=True)
    dset.replace({'Dport_num': [np.nan]}, 0, inplace=True)

    srcPort_cond = [(dset['Sport_num'] == -1),
                    (dset['Sport_num'] >= 0) & (dset['Sport_num'] <= 1023),
                    (dset['Sport_num'] >= 1024) & (dset['Sport_num'] <= 49151),
                    (dset['Sport_num'] > 49151)]
    dstPort_cond = [(dset['Dport_num'] == -1),
                    (dset['Dport_num'] >= 0) & (dset['Dport_num'] <= 1023),
                    (dset['Dport_num'] >= 1024) & (dset['Dport_num'] <= 49151),
                    (dset['Dport_num'] > 49151)]
    
    port_choices = ['none','well-known','registered','dynamic']
    port_choices = [-1,0,1,2] # -1 = none, 0 =well-known, 1=registered, 2=dynamic
    dset['Sport_type'] = np.select(srcPort_cond, port_choices)
    dset['Dport_type'] = np.select(dstPort_cond, port_choices)
    
    dset.drop(columns=['Sport', 'Dport', 'Dport_num', 'Sport_num'], inplace=True)
    return dset

def fixIPs(addr):
    ip_patterns = [r"^10\.", r"^172\.(1[6-9]2\d|3[01])\.", r"^192\.168\.", "^127\.", r"^fe80::"]
    for p in ip_patterns:
        if re.match(p, addr):
            return 1
    return 0

def preprocess(dset: pd.DataFrame, filter, flow_file: FlowFile, test_size):
    # Riduzione del numero di righe
    dset['seed'] = (np.random.uniform(0,1,len(dset)))
    dset = dset[dset['seed'] < filter]

    # Assegnazione date e label
    dset['Date'] = flow_file.date
    dset.drop(columns=['StartTime', 'seed'], inplace=True)
    # dset['Label_original'] = dset['Label']
    if flow_file.name in ["background", "normal"]:
        dset['Label'] = 0
    elif flow_file.name in malware_list:
        dset['Label'] = 1
    
    # Split train | test
    # dset['seed'] = (np.random.uniform(0,1,len(dset)))
    # dset['is_test'] = np.where(dset['seed'] <= test_size, True, False)

    dset = fixPorts(dset)
    dset = fixValues(dset)
    dset.loc[:, "SrcPrivateIP"] = dset["SrcAddr"].apply(fixIPs)
    dset.loc[:, "DstPrivateIP"] = dset["DstAddr"].apply(fixIPs)
    dset.drop(columns=["SrcAddr", "DstAddr"], inplace=True)
    return dset

def handleCategorical(dset: pd.DataFrame):
    # dset['Nature'] = np.where(dset['Label'].str.contains("BENIGN"), 0, 1)

    for c in dset.columns:
        if c in ['State', 'Flgs', 'Proto', 'Dir']:
            dset[c + "-f"] = pd.factorize(dset[c])[0]
        else:
            pass
    dset.drop(columns=['State', 'Flgs', 'Proto', 'Dir'], inplace=True)
    # dset['Label_cat'] = pd.factorize(dset['Label'])[0]
    return dset

def print_metrics(metrics):
    m = pd.DataFrame(metrics)
    m.set_index('Date', inplace=True)

    print(m.to_markdown(numalign="center", stralign="center"))
    print()

malware_list = ["trickbot", "dridex", "wannacry", "artemis"]
test_size = 0.3
filter_malicious = 1
filter_normal = 1

normal_list = [
    # FlowFile("2013", "12", "17", "normal", "1"),
    FlowFile("2016", "09", "13", "normal", "2"),
    FlowFile("2016", "09", "13", "normal", "3"),
    FlowFile("2016", "09", "13", "normal", "4"),
    FlowFile("2017", "07", "03", "normal", "5"),
    FlowFile("2017", "07", "23", "normal", "6"),
    FlowFile("2017", "09", "05", "normal", "7"),
    FlowFile("2017", "04", "30", "normal", "8"),
    FlowFile("2017", "05", "02", "normal", "9"),
    FlowFile("2017", "05", "08", "normal", "10"),
    FlowFile("2017", "04", "18", "normal", "11"),
    FlowFile("2017", "04", "19", "normal", "12"),
    FlowFile("2017", "04", "25", "normal", "13"),
    FlowFile("2017", "04", "28", "normal", "14"),
    FlowFile("2017", "04", "30", "normal", "15"),
    FlowFile("2017", "05", "01", "normal", "16"),
    FlowFile("2017", "05", "01", "normal", "17"),
    FlowFile("2017", "05", "01", "normal", "18"),
    FlowFile("2017", "05", "01", "normal", "19"),
    # FlowFile("2013", "12", "17", "normal", "20"),
    FlowFile("2017", "05", "02", "normal", "21"),
    FlowFile("2018", "05", "07", "normal", "22")
]

malicious_list = [
    # FlowFile("2017", "3", "30", "trickbot", "1"),  
    # FlowFile("2017", "3", "30", "trickbot", "2"),  
    # FlowFile("2017", "3", "29", "trickbot", "3"),  
    # FlowFile("2017", "3", "30", "trickbot", "4"),  
    # FlowFile("2017", "3", "30", "trickbot", "5"),  
    FlowFile("2017", "04", "12", "trickbot", "6"),  
    FlowFile("2017", "04", "12", "trickbot", "7"),  
    FlowFile("2017", "04", "17", "trickbot", "8"),  
    FlowFile("2017", "05", "15", "trickbot", "9"),  
    FlowFile("2017", "06", "07", "trickbot", "10"), 
    FlowFile("2017", "06", "24", "trickbot", "11"), 
    FlowFile("2017", "06", "24", "trickbot", "12"), 
    FlowFile("2017", "06", "24", "trickbot", "13"), 
    FlowFile("2017", "06", "24", "trickbot", "14"), 
    # FlowFile("2018", "01", "30", "trickbot", "15"), 
    # FlowFile("2018", "01", "30", "trickbot", "16"), 
    # FlowFile("2021", "07", "30", "trickbot", "17"),
    # FlowFile("2018", "03", "27", "trickbot", "18"),   

    # FlowFile("2015", "03", "12", "dridex", "1"),
    # FlowFile("2016", "02", "12", "dridex", "2"),
    # FlowFile("2017", "02", "27", "dridex", "4"),
    # FlowFile("2017", "02", "27", "dridex", "5"),
    FlowFile("2017", "04", "17", "dridex", "6"),
    FlowFile("2017", "04", "18", "dridex", "7"),
    FlowFile("2017", "04", "18", "dridex", "8"),
    FlowFile("2017", "05", "15", "dridex", "10"),
    FlowFile("2017", "05", "16", "dridex", "9"),
    FlowFile("2017", "06", "24", "dridex", "12"),
    FlowFile("2017", "05", "15", "dridex", "11"),
    # FlowFile("2018", "01", "29", "dridex", "13"),
    # FlowFile("2018", "01", "30", "dridex", "14"),
    FlowFile("2018", "04", "13", "dridex", "15"),

    FlowFile("2017", "06", "24", "artemis", "1"),
    FlowFile("2017", "08", "14", "artemis", "2"),
    FlowFile("2017", "08", "01", "artemis", "3"),
    FlowFile("2017", "08", "15", "artemis", "4"),
    FlowFile("2017", "08", "16", "artemis", "5"),

    FlowFile("2017", "05", "14", "wannacry", "1"),
    FlowFile("2017", "05", "14", "wannacry", "2"),
    FlowFile("2017", "05", "15", "wannacry", "3"),
    FlowFile("2017", "05", "15", "wannacry", "4"),
    FlowFile("2017", "05", "16", "wannacry", "5"),
    FlowFile("2017", "06", "24", "wannacry", "6"),
    FlowFile("2017", "07", "11", "wannacry", "7"),
    FlowFile("2017", "07", "11", "wannacry", "8"),
    FlowFile("2017", "07", "11", "wannacry", "9"),
    FlowFile("2017", "07", "22", "wannacry", "10"),
    FlowFile("2017", "07", "11", "wannacry", "11"),
    FlowFile("2017", "07", "13", "wannacry", "12"),
    FlowFile("2017", "07", "11", "wannacry", "13"),
    FlowFile("2017", "07", "13", "wannacry", "14"),
    FlowFile("2017", "07", "13", "wannacry", "15")
    ]

needed_features = ['Dur', 'SrcDur', 'DstDur', 'sTos', 'dTos',  'dTtl',
           'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'Label',
           'SrcBytes', 'DstBytes', 'TotAppByte', 'SAppBytes', 'DAppBytes', 'Load',
           'SrcLoad', 'DstLoad', 'Rate', 'SrcRate', 'DstRate', 'DstTCPBase', 'SrcTCPBase', 
            'TcpRtt', 'SynAck', 'AckDat', 'sMaxPktSz', 'sMinPktSz', 'dMaxPktSz', 'dMinPktSz', 
           'Sport_type', 'Dport_type', 'Flgs-f', 'Dir-f', 'Proto-f', 'State-f', 'Loss', 'Date']

all_dsets = pd.DataFrame()

print("Reading MALICIOUS...")
for m in malicious_list:
    # print(f"Reading file {m.filename}")
    m_file = f"malicious/{m.name}/{m.filename}"
    m_dset = pd.read_csv(m_file, dtype={'Sport':'object', 'Dport':'object'})
    m_dset = preprocess(m_dset, filter_malicious, m, test_size)
    all_dsets = pd.concat([all_dsets, m_dset])
    
print("Reading NORMAL...")
for n in normal_list:
    # print(f"Reading file {n.filename}")
    n_file = f"normal/{n.filename}"
    n_dset = pd.read_csv(n_file, dtype={'Sport':'object', 'Dport':'object'})
    n_dset = preprocess(n_dset, filter_normal, n, test_size)
    all_dsets = pd.concat([all_dsets, n_dset])

all_dsets = handleCategorical(all_dsets)
all_dsets = all_dsets[needed_features]
all_dsets = all_dsets.reset_index(drop=True)
all_dsets['Date'] = pd.to_datetime(all_dsets['Date'])

# Se ci sono ancora valori nulli, sono da rimuovere
# print(all_dsets.isna().any().any())
# print(all_dsets.columns[all_dsets.isna().any()])

# -------------------- TRAINING --------------------

with open("performances/tmp.txt", "a") as f:
    f.write("START\n\n")

all_results = []
for i in range(1,6):
    print(f"Test {i}")
    results_standard = ensemble_cl(all_dsets, test_size, botnet="all")
    print_metrics(results_standard)
    all_results.append(results_standard)

mean_results = {}
metrics = ["Precision", "F1", "TPR", "TNR"]
# metrics = ['TPR']
for m in metrics:
    values = [run[m] for run in all_results]
    mean_results[m] = np.mean(values, axis=0).tolist()

for key, value in mean_results.items():
    floored_list = [decimal_floor(n, 3) for n in value]
    mean_results[key] = floored_list

print(f"Valori medi: {mean_results}")

# -------------------- GRAFICI --------------------
pendleblue="#1F8FFF"
pendlegreen="#32c63c"
pendlered="#B11616"
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(mean_results['Precision'], marker='o', color=pendlegreen)
ax1.plot(mean_results['F1'], marker='o', color=pendleblue)
ax1.plot(mean_results['TNR'], marker='o', color='cyan')
ax1.plot(mean_results['TPR'], marker='o', color=pendlered)
ax1.legend(['Precision', 'F1', 'TNR', 'TPR'])
# ax1.legend(['TPR'])
ax1.set_ylabel('Performance')
ax1.grid(axis = 'y')
ax1.set_xticks(range(len(results_standard['Date'])))
ax1.set_xticklabels(results_standard['Date'])
ax1.set_title("ALL BOTNETS")

# ax2.plot(results_downsample['Precision'], marker='o', color=pendleyellow)
# ax2.plot(results_downsample['Recall'], marker='o', color='red')
# ax2.plot(results_downsample['F1'], marker='o', color=pendleblue)
# ax2.legend(['Precision', 'Recall', 'F1'])
# ax2.grid(axis = 'y')
# ax2.set_title('')

plt.tight_layout()

plt.show()
