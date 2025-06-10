from alliGATOR import *
import itertools
import csv
import numpy as np

def how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result):
    
    increase_list = []
    for k, ds, a, g in itertools.product(K, DS, A, guidance):

        
        case_1 = (k, ds, a, g, 50)
        case_2 = (k, ds, a, g, 100)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_guidance_affect_the_result(K, DS, A, MC, dict_result):
    
    increase_list = []
    for k, ds, a, mc in itertools.product(K, DS, A, MC):
        
        case_1 = (k, ds, a, 0, mc)
        case_2 = (k, ds, a, 4.5, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_attention_affect_the_result(K, DS, guidance, MC, dict_result):
    
    increase_list = []
    for k, ds, g, mc in itertools.product(K, DS, guidance, MC):
        
        case_1 = (k, ds, 31, g, mc)
        case_2 = (k, ds, 331, g, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

# ----------------------------------------------------------------------

K = [15]
DS = [4]
A = [1, 31, 331]
guidance = [0, 0.5, 4.5]
mc_simulation = [10, 50, 100]

valid_combinations = [
    (k, ds, a, g, mc)
    for k, ds, a, g, mc in itertools.product(K, DS, A, guidance, mc_simulation)
]
print(f"The number of possible combinations is {len(valid_combinations)}")

# """ This is for reading generating the results for all combinations
check = {"DS4A1": 407, 
         "DS4A31": 218, 
         "DS4A331": 311}

# results = []
# for combo in valid_combinations:
#     k, ds, a, g, mc = combo
#     print("-------------------------------------------------------")
#     print(f"Starting with combination K {k}, DS {ds}, A {a}, G {g}, MC {mc}")
#     NAG = alliGATOR(f"./wandb/Synthetic_K{k}_node/multinomial_diffusion/multistep/DS{ds}A{a}", check[f"DS{ds}A{a}"], MC = mc, name = f"K{k}DS{ds}A{a}", lambda_guidance=g, 
#                                sample_numbers=945, tuning = True, anomaly_type="node_anomaly") 
#     precision, recall, prauc, avg_precision = NAG.get_PR_AUC(NAG.get_true_anomaly_label_tf_theft(), NAG.get_id_theft_prediction(), title_PR_type="Node Anomaly")
#     results.append({'params': combo, 'pr-auc': prauc, 'average-precision' : avg_precision})

# with open(f"Alligator_Output_node_anomaly/Node_anomaly_experiment.pkl", "wb") as f:
#     pickle.dump(results, f)
# """

# ----------------------------------------------------------------------

with open(f"Alligator_Output_node_anomaly/Node_anomaly_experiment.pkl", 'rb') as f:
    results = pickle.load(f)

sorted_results = sorted(results, key=lambda x: x['pr-auc'], reverse=True)
best_result = sorted_results[0]

print(f"Best parameters: {best_result['params']}")
print(f"Best score (R): {best_result['pr-auc']}")

sorted_results = sorted(results, key=lambda x: x['average-precision'], reverse=True)
best_result = sorted_results[0]

print(f"Best parameters: {best_result['params']}")
print(f"Best score (R): {best_result['average-precision']}")

dict_result = {}
for r in results:
    dict_result[r["params"]] = r["average-precision"]

# how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result)
how_much_does_the_guidance_affect_the_result(K, DS, A, mc_simulation, dict_result)
# how_much_does_the_attention_affect_the_result(K, DS, guidance, mc_simulation, dict_result)
