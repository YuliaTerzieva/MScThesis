from alliGATOR import *
import itertools
import csv
import numpy as np


K = [7, 15]
DS = [10, 20]
A = [1, 21]
guidance = [0, 0.5, 4.5]
mc_simulation = [10, 100, 1000]

valid_combinations = [
    (k, ds, a, g, mc)
    for k, ds, a, g, mc in itertools.product(K, DS, A, guidance, mc_simulation)
    if not (k == 7 and ds == 20)
]

check = {"K7DS10A1" : 979, 
         "K7DS10A21" : 879, 
         "K15DS10A1" : 559, 
         "K15DS10A21" : 619, 
         "K15DS20A1" : 919, 
         "K15DS20A21" : 539}

results = []

for combo in valid_combinations:
    k, ds, a, g, mc = combo
    print("-------------------------------------------------------")
    print(f"Starting with combination K {k}, DS {ds}, A {a}, G {g}, MC {mc}")
    EAG = alliGATOR(f"./wandb/Edge_classification_K{k}/multinomial_diffusion/multistep/K{k}DS{ds}A{a}", check[f"K{k}DS{ds}A{a}"], MC = mc, name = f"K{k}DS{ds}A{a}", lambda_guidance=g, 
                    sample_numbers=2186, anomaly_type="edge_anomaly") 
    prauc = EAG.get_PR_AUC(EAG.get_true_anomaly_labels_for_edge_cls(), EAG.get_edge_cls_anomaly(), title_PR_type = "Edge anomaly")
    results.append({'params': combo, 'pr-auc': prauc})

with open(f"Alligator_Output_edge_anomaly/edge_anomaly_experiment.pkl", "wb") as f:
    pickle.dump(results, f)