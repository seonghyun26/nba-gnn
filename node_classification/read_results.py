

import os
import yaml
import numpy as np
import pandas as pd

datasets = ['cora', 'citeseer', 'pubmed', 'texas', 'wisconsin', 'cornell']
models=['GCN','nbtm_GCN','PE_nbtm_GCN','SAGE','nbtm_SAGE','PE_nbtm_SAGE','GAT','nbtm_GAT','PE_nbtm_GAT','gated_GCN','nbtm_gated_GCN','PE_nbtm_gated_GCN']
df = pd.DataFrame(columns=datasets)

for model in models:
    model_name = model.replace('-', '_').replace('+', '_')
    results = []
    for dataset in datasets:
        val_metrics = []
        metric_name = 'accuracy'
        for num_layers in [3,4,5]:
            filename = f'./results/{dataset}/{model_name}_l{num_layers}_01/metrics.yaml'
            if os.path.exists(filename):
                with open(filename) as file:
                    metrics = yaml.safe_load(file)
                val_metric_mean = metrics[f'val {metric_name} mean']
                val_metrics.append(val_metric_mean)
            else: 
                val_metrics.append(0)
       
        best_num_layers = np.argmax(val_metrics) + 3
        filename = f'./results/{dataset}/{model_name}_l{best_num_layers}_01/metrics.yaml'
        if os.path.exists(filename):
            with open(filename) as file:
                metrics = yaml.safe_load(file)

            test_metric_mean = metrics[f'test {metric_name} mean']
            test_metric_std = metrics[f'test {metric_name} std']

            string = f'{test_metric_mean *100 :.2f} \u00B1 {test_metric_std *100:.2f}'
        else:
            string=''  
        results.append(string)
    df.loc[model] = results

print(df)

