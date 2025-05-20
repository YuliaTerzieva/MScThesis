from alliGATOR import *
import itertools
import csv
import numpy as np

def how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result):
    
    increase_list = []
    for k, ds, a, g in itertools.product(K, DS, A, guidance):
        if (k == 7 and ds == 20):
            continue
        
        case_1 = (k, ds, a, g, 100)
        case_2 = (k, ds, a, g, 1000)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_guidance_affect_the_result(K, DS, A, MC, dict_result):
    
    increase_list = []
    for k, ds, a, mc in itertools.product(K, DS, A, MC):
        if (k == 7 and ds == 20):
            continue
        
        case_1 = (k, ds, a, 0, mc)
        case_2 = (k, ds, a, 0.5, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_k_affect_the_result(DS, A, guidance, MC, dict_result):
    
    increase_list = []
    for ds, a, g, mc in itertools.product(DS, A, guidance, MC):
        if (ds == 20):
            continue
        
        case_1 = (7, ds, a, g, mc)
        case_2 = (15, ds, a, g, mc)
        increase_list.append(dict_result[case_2] - dict_result[case_1])
    
    print(increase_list)
    print("Mean = ", round(np.mean(increase_list), 3),"STD = ",  round(np.std(increase_list), 3))
    return sum(increase_list)/len(increase_list)

def how_much_does_the_attention_affect_the_result(K, DS, guidance, MC, dict_result):
    
    increase_list = []
    for k, ds, g, mc in itertools.product(K, DS, guidance, MC):
        if (k == 7 and ds == 20):
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
    a = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array([7, 15])  # K
    y = np.array([10, 20])  # DS
    xpos, ypos = np.meshgrid(x, y, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dz = []
    for k in x:
        for ds in y:
            if k == 7 and ds == 20 : 
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

# for combo in valid_combinations:
#     k, ds, a, g, mc = combo
#     print("-------------------------------------------------------")
#     print(f"Starting with combination K {k}, DS {ds}, A {a}, G {g}, MC {mc}")
#     EAG = alliGATOR(f"./wandb/Edge_classification_K{k}/multinomial_diffusion/multistep/K{k}DS{ds}A{a}", check[f"K{k}DS{ds}A{a}"], MC = mc, name = f"K{k}DS{ds}A{a}", lambda_guidance=g, 
#                     sample_numbers=2186, previously_sampled_model_filename=f"Alligator_Output_edge_anomaly/K{k}DS{ds}A{a}_mc{mc}_guidance{int(g*10)}.pkl", anomaly_type="edge_anomaly")
#     precision, recall, prauc = EAG.get_PR_AUC(EAG.get_true_anomaly_labels_for_edge_cls(), EAG.get_edge_cls_anomaly(), title_PR_type = "Edge anomaly")
#     results.append({'params': combo, 'pr-auc': prauc})

# with open(f"Alligator_Output_edge_anomaly/edge_anomaly_experiment.pkl", "wb") as f:
#     pickle.dump(results, f)

with open(f"Alligator_Output_edge_anomaly/edge_anomaly_experiment.pkl", "rb") as f:
    results = pickle.load(f)

sorted_results = sorted(results, key=lambda x: x['pr-auc'], reverse=True)
best_result = sorted_results[0]

print(f"Best parameters: {best_result['params']}")
print(f"Best score (R): {best_result['pr-auc']}")


dict_result = {}
for r in results:
    dict_result[r["params"]] = r["pr-auc"]

# how_much_does_mc_affect_the_result(K, DS, A, guidance, dict_result)
# how_much_does_the_guidance_affect_the_result(K, DS, A, mc_simulation, dict_result)
# how_much_does_the_k_affect_the_result(DS, A, guidance, mc_simulation, dict_result)
# how_much_does_the_attention_affect_the_result(K, DS, guidance, mc_simulation, dict_result)
# k_and_ds_result(dict_result)