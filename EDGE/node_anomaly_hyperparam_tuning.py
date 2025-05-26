from alliGATOR import *
import itertools
import csv
import numpy as np

def how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result):
    
    increase_list = []
    for k, ds, a, g in itertools.product(K, DS, A, guidance):
        if (k == 10 and ds == 15):
            continue
        
        case_1 = (k, ds, a, g, 10)
        case_2 = (k, ds, a, g, 100)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_guidance_affect_the_result(K, DS, A, MC, dict_result):
    
    increase_list = []
    for k, ds, a, mc in itertools.product(K, DS, A, MC):
        if (k == 10 and ds == 15):
            continue
        
        case_1 = (k, ds, a, 0.5, mc)
        case_2 = (k, ds, a, 4.5, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_k_affect_the_result(DS, A, guidance, MC, dict_result):
    
    increase_list = []
    for ds, a, g, mc in itertools.product(DS, A, guidance, MC):
        if (ds == 15):
            continue
        
        case_1 = (10, ds, a, g, mc)
        case_2 = (15, ds, a, g, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_attention_affect_the_result(K, DS, guidance, MC, dict_result):
    
    increase_list = []
    for k, ds, g, mc in itertools.product(K, DS, guidance, MC):
        if (k == 10 and ds == 15):
            continue
        
        case_1 = (k, ds, 1, g, mc)
        case_2 = (k, ds, 21, g, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def k_and_ds_result(dict_result):
    
    mc = 1000
    g = 0.5
    a = 21

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array([10, 15])  # K
    y = np.array([8, 10, 15])  # DS
    xpos, ypos = np.meshgrid(x, y, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dz = []
    for k in x:
        for ds in y:
            if k == 10 and ds == 15 : 
                dz.append(0)
                continue
            dz.append(dict_result[(k, ds, a, g, mc)])

    # Bar dimensions
    dx = dy = 1.0

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    ax.set_xlabel('K')
    ax.set_ylabel('diffusion steps')
    ax.set_zlabel('PR-AUC')

    ax.set_xticks(x)
    ax.set_yticks(y)

    ax.set_title('alliGATOR hyperparameter tuning', fontsize=14, fontweight='bold')

    plt.show()

# ----------------------------------------------------------------------

K = [10, 15]
DS = [8, 10, 15]
A = [1, 21]
guidance = [0, 0.5, 4.5]
mc_simulation = [10, 100, 1000]

valid_combinations = [
    (k, ds, a, g, mc)
    for k, ds, a, g, mc in itertools.product(K, DS, A, guidance, mc_simulation)
    if not (k == 10 and ds == 15)
]

""" This is for reading generating the results for all combinations
check = {"K15DS8A1" : 659, 
         "K15DS8A21" : 799, 
         "K15DS10A1" : 399, 
         "K15DS10A21" : 619, 
         "K15DS15A1" : 1609, 
         "K15DS15A21" :799, 
         "K10DS8A1" : 869, 
         "K10DS8A21": 799, 
         "K10DS10A1": 439, 
         "K10DS10A21": 619}

results = []
for combo in valid_combinations:
    k, ds, a, g, mc = combo
    print("-------------------------------------------------------")
    print(f"Starting with combination K {k}, DS {ds}, A {a}, G {g}, MC {mc}")
    NAG = alliGATOR(f"./wandb/Id_theft_K{k}/multinomial_diffusion/multistep/K{k}DS{ds}A{a}", check[f"K{k}DS{ds}A{a}"], MC = mc, name = f"K{k}DS{ds}A{a}", lambda_guidance=g, 
                               sample_numbers=1000, anomaly_type="node_anomaly", previously_sampled_model_filename=f"Alligator_Output_node_anomaly/K{k}DS{ds}A{a}_mc{mc}_guidance{int(g*10)}.pkl") 
    precision, recall, prauc = NAG.get_PR_AUC(NAG.get_true_anomaly_label_tf_theft(), NAG.get_id_theft_prediction(), title_PR_type="Node Anomaly")
    results.append({'params': combo, 'pr-auc': prauc})

with open(f"Alligator_Output_node_anomaly/Node_anomaly_experiment.pkl", "wb") as f:
    pickle.dump(results, f)
"""

# ----------------------------------------------------------------------

with open(f"Alligator_Output_node_anomaly/Node_anomaly_experiment.pkl", 'rb') as f:
    results = pickle.load(f)

sorted_results = sorted(results, key=lambda x: x['pr-auc'], reverse=True)
best_result = sorted_results[0]

print(f"Best parameters: {best_result['params']}")
print(f"Best score (R): {best_result['pr-auc']}")


# dict_result = {}
# for r in results:
#     dict_result[r["params"]] = r["pr-auc"]

# how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result)
# how_much_does_the_guidance_affect_the_result(K, DS, A, mc_simulation, dict_result)
# how_much_does_the_k_affect_the_result(DS, A, guidance, mc_simulation, dict_result)
# how_much_does_the_attention_affect_the_result(K, DS, guidance, mc_simulation, dict_result)
# k_and_ds_result(dict_result)

# all_conditions_met = True
# for k, ds, a, g in itertools.product(DS, A, guidance):
#     if (k == 10 and ds == 15):
#         continue
    
    
#     key_a1 = (10, ds, a, g, 10)
#     key_a21 = (15, ds, a, g, 100)


#     if key_a1 in dict_result and key_a21 in dict_result:
#         if dict_result[key_a1] >= dict_result[key_a21]:
#             print(f"Condition failed for combination {key_a1} vs {key_a21}: "
#                 f"{dict_result[key_a1]} >= {dict_result[key_a21]}")
#             all_conditions_met = False
#     else:
#         print(f"Missing data for combination {key_a1} or {key_a21}")

# if all_conditions_met:
#     print("Condition holds for all valid combinations!")
# else:
#     print("Condition failed for one or more combinations.")


# ----------------------------------------------------------------------

# # Define CSV filename
# csv_filename = 'hyperparameter_results.csv'

# # Write to CSV
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # Write header
#     writer.writerow(['K', 'diffusion steps', 'Attention', 'Guidance', 'MC Simulation', 'Result'])

#     # Write rows
#     for params, result in dict_result.items():
#         writer.writerow(list(params) + [round(result, 3)])

# print(f"Results saved to {csv_filename}")
# ----------------------------------------------------------------------

# NAG = alliGATOR(f"./wandb/Id_theft_K{15}/multinomial_diffusion/multistep/K{15}DS{8}A{21}", check[f"K{15}DS{8}A{21}"], MC = 1000, name = f"K{15}DS{8}A{21}", lambda_guidance=4.5, 
#                 sample_numbers=1000, previously_sampled_model_filename=f"Alligator_Output/K{15}DS{8}A{21}_mc{1000}_guidance{int(4.5*10)}.pkl") 
# prauc = NAG.get_PR_AUC(NAG.get_true_anomaly_label_tf_theft(), NAG.get_id_theft_prediction(), title_PR_type="Node Anomaly")



