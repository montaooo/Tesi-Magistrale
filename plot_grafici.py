import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

aut_1 = [
    [0.9675, 0.9679, 0.9670],
    [0.4877, 0.6049, 0.7095],
    [0.5229, 0.6363, 0.7419]
]

aut_2 = [
    [0.9664, 0.9623, 0.9652],
    [0.4738, 0.3570, 0.3790],
    [0.5129, 0.3745, 0.3990]
]

aut_3 = [
    [0.9670, 0.9633, 0.9685],
    [0.4625, 0.3307, 0.4257],
    [0.4960, 0.3435, 0.4436]
]

aut_4 = [
    [0.9689, 0.9689, 0.9685],
    [0.3971, 0.5987, 0.8576],
    [0.4387, 0.6403, 0.8827]
]

aut_5 = [
    [0.9676, 0.9648, 0.9646],
    [0.5258, 0.4304, 0.3798],
    [0.5611, 0.4414, 0.3817]
]

aut_6 = [
    [0.9704, 0.9419, 0.9466],
    [0.4455, 0.4941, 0.5804],
    [0.4884, 0.5189, 0.5908]
]

aut_7 = [
    [0.9664, 0.9649, 0.9683],
    [0.4136, 0.4878, 0.5748],
    [0.4505, 0.4827, 0.5979]
]

aut_8 = [
    [0.9679, 0.9677, 0.9733],
    [0.4761, 0.5012, 0.5636],
    [0.5137, 0.5065, 0.5988]
]

aut_9 = [
    [0.9692, 0.9682, 0.9684],
    [0.3452, 0.5740, 0.8235],
    [0.3844, 0.6135, 0.8607]
]

mean_aut = []
for metric_values in zip(aut_1, aut_2, aut_3, aut_4, aut_5, aut_6, aut_7, aut_8, aut_9):
    mean_aut.append(np.mean(metric_values, axis=0).tolist())

print(mean_aut)

concept_drift = {
    'Date': ['2017-04-01', '2017-05-01', '2017-06-01', '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01'],
    'Precision': [0.999, 0.633, 1.0, 0.997, 1.0, 0.985, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999],
    'F1': [0.999, 0.74, 0.963, 0.176, 0.948, 0.117, 0.103, 0.138, 0.618, 0.816, 0.0, 0.468, 0.396],
    'TNR': [0.999, 0.797, float("nan"), 0.998, float("nan"), 0.986, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0.998],
    'TPR': [0.999, 0.887, 0.932, 0.104, 0.904, 0.061, 0.054, 0.076, 0.553, 0.73, 0.0, 0.354, 0.312]
}

cl = {
    'Date': ['2017-04-01', '2017-05-01', '2017-06-01', '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01'],
    'Precision': [0.999, 0.631, 1.0, 0.998, 1.0, 0.984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'F1': [0.999, 0.737, 0.999, 0.903, 0.559, 0.347, 0.015, 0.245, 0.383, 0.53, 0.0, 0.609, 0.301],
    'TNR': [0.999, 0.793, float("nan"), 0.992, float("nan"), 0.983, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 1.0],
    'TPR': [0.999, 0.887, 0.999, 0.826, 0.546, 0.31, 0.007, 0.147, 0.304, 0.381, 0.0, 0.499, 0.205]
}

mu = {
    'Date': ['2017-04-01', '2017-05-01', '2017-06-01', '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01'],
    'Precision': [0.999, 0.632, 1.0, 0.998, 1.0, 0.981, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'F1': [0.999, 0.735, 0.999, 0.871, 0.548, 0.291, 0.231, 0.407, 0.406, 0.485, 0.285, 0.686, 0.463],
    'TNR': [0.999, 0.795, float("nan"), 0.993, float("nan"), 0.993, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0.999],
    'TPR': [0.999, 0.888, 0.999, 0.767, 0.536, 0.222, 0.153, 0.316, 0.347, 0.359, 0.245, 0.584, 0.373]
}

plot_data = {
    "no_retrain": concept_drift,
    "cl": cl,
    "cl_mu": mu
}
for key in plot_data:
    plot_data[key] = pd.DataFrame(plot_data[key])
    plot_data[key]['Date'] = pd.to_datetime(plot_data[key]['Date'])

pendleblue="#1264B7"
pendlegreen="#47a91d"
pendlered="#901212"
labels_info = [
    ('no_retrain', pendlered),
    ('cl', pendleblue),
    ('cl_mu', pendlegreen),
]

metrics = ['Precision', 'F1', 'TPR']

# fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 18), sharex=True)

for metric in metrics:
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    
    for strategy_key, color in labels_info:
        df = plot_data[strategy_key]
        mask = df[metric].notna()
        plot_df = df[mask]
        
        plt.plot(plot_df['Date'], plot_df[metric], 
                 label=strategy_key,
                 color=color, 
                 linewidth=2, 
                 marker='o', 
                 markersize=5)
    
    
    plt.ylabel(metric, fontsize=16)
    plt.xlabel('Data', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show() 


# Istogramma dei valori AUT
no_retrain_vals = [mean_aut[0][0], mean_aut[1][0], mean_aut[2][0]]
cl_vals = [mean_aut[0][1], mean_aut[1][1], mean_aut[2][1]]
cl_mu_vals = [mean_aut[0][2], mean_aut[1][2], mean_aut[2][2]]
metrics_label = ["Precision", "TPR", "F1"]
labels = ['no_retrain', 'cl', 'cl_mu']
colors = [pendlered, pendleblue, pendlegreen]

x = np.arange(3)
width = 0.25

fig, ax = plt.subplots(figsize=(15,10))

graph1 = ax.bar(x - width, no_retrain_vals, width, label='no_retrain', color=pendlered, edgecolor='black', alpha=0.8)
graph2 = ax.bar(x, cl_vals, width, label='cl', color=pendleblue, edgecolor='black', alpha=0.8)
graph3 = ax.bar(x + width, cl_mu_vals, width, label='cl_mu', color=pendlegreen, edgecolor='black', alpha=0.8)

ax.set_ylabel("AUT", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics_label, fontsize=16)
ax.set_ylim(0, 1.1)
def add_value_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

add_value_labels(graph1)
add_value_labels(graph2)
add_value_labels(graph3)
ax.legend(fontsize=14)
ax.set_xlim(-0.8, 2.8)
plt.tight_layout()
plt.show()