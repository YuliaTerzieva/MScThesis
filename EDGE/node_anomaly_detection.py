from alliGATOR import *


all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)
all_avg_precision = []


for n in range(1):

    node_anomaly_gator = alliGATOR(f"./wandb/Synthetic_K15_node/multinomial_diffusion/multistep/2025-06-14_23-42-37", 197, MC = 100, name = f"DS{4}A{31}_no_dtst_", lambda_guidance=4.5, 
                    sample_numbers=945, previously_sampled_model_filename="./Alligator_Output_node_anomaly/DS4A31_no_dtst__mc100_guidance45.pkl", anomaly_type="node_anomaly", seed = n, tuning = False) 

    precision, recall, auc_precision_recall, avg_precision = node_anomaly_gator.get_PR_AUC(node_anomaly_gator.get_true_anomaly_label_tf_theft(), node_anomaly_gator.get_id_theft_prediction(), title_PR_type="Node Anomaly")
    # plt.plot(recall, precision, ".-")
    # plt.show()
    precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    
    all_precision.append(precision_interp)
    all_avg_precision.append(avg_precision)

all_precision = np.vstack(all_precision)
mean_precision = all_precision.mean(axis=0)
std_precision = all_precision.std(axis=0)

# Plot mean curve with shaded std area
plt.figure(figsize=(5, 5))
plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AP = {np.mean(all_avg_precision):.5f})", color = "#4A21A8")
plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.5, color = "#7DC3F6")

# Baseline (random)
labels = node_anomaly_gator.get_true_anomaly_label_tf_theft()
positive_ratio = labels.count(1)/len(labels)
plt.hlines(positive_ratio, xmin=0, xmax=1, color="#F1993A", label='Baseline')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Node anomaly on Synthetic dataset \nMean Precision-Recall Curve with Std (alliGATOR)")
plt.legend()
# plt.savefig("NAD-Gator-AUPRC")
plt.show()
#------------------------->>>

# node_anomaly_gator = alliGATOR("./wandb/Id_theft/multinomial_diffusion/multistep/2025-04-27_19-36-11", 319, MC = 100, name = "id_theft", lambda_guidance=4.5, 
#                           sample_numbers=1000)
# node_anomaly_gator = alliGATOR("./wandb/Id_theft_2/multinomial_diffusion/multistep/2025-04-27_22-09-16", 919, MC = 500, name = "id_theft_2", lambda_guidance=4.5, 
#                            sample_numbers=1000, previously_sampled_model_filename="Alligator_Output/id_theft_2_mc500_guidance45.pkl")

# node_anomaly_gator = alliGATOR(f"./wandb/Id_theft_K{15}/multinomial_diffusion/multistep/K{15}DS{8}A{21}", 799, MC = 1000, name = f"K{15}DS{8}A{21}", lambda_guidance=4.5, 
#                     sample_numbers=1000, previously_sampled_model_filename=f"Alligator_Output/K{15}DS{8}A{21}_mc{1000}_guidance{int(4.5*10)}.pkl") 

# node_anomaly_gator = alliGATOR(f"./wandb/Id_theft_K{15}/multinomial_diffusion/multistep/K{15}DS{8}A{21}", 799, MC = 1000, name = f"K{15}DS{8}A{21}", lambda_guidance=-1, 
#                     sample_numbers=1000) 
# node_anomaly_gator = alliGATOR(f"./wandb/Id_theft_K{15}/multinomial_diffusion/multistep/K{15}DS{8}A{21}", 799, MC = 1000, name = f"K{15}DS{8}A{21}", lambda_guidance=4.5, 
#                     sample_numbers=1000, previously_sampled_model_filename=f"Alligator_Output_node_anomaly/K{15}DS{8}A{21}_mc{1000}_guidance{45}.pkl") 


# node_anomaly_gator.get_PR_AUC(node_anomaly_gator.get_true_anomaly_label_tf_theft(), node_anomaly_gator.get_id_theft_prediction(), title_PR_type="Node Anomaly")

# node_anomaly_gator.plot_active_edges_and_nodes()
# node_anomaly_gator.plot_edge_distribution_violin_boxplots(node_anomaly_gator.get_per_edge_type_probability_list(only_originla_edges=True), "original edges only")
# node_anomaly_gator.plot_edge_distribution_violin_boxplots(node_anomaly_gator.get_per_edge_type_probability_list(only_originla_edges=True, node_degree_adjusted=True), "original edges only")
# node_anomaly_gator.plot_edge_distribution_violin_boxplots(node_anomaly_gator.get_per_edge_type_probability_list(only_originla_edges=False), "all generated edges")
# node_anomaly_gator.plot_edge_distribution_violin_boxplots(node_anomaly_gator.get_per_edge_type_probability_list(only_originla_edges=False, node_degree_adjusted=True), "all generated edges")

# --------->  which are the anomalous graphs?

# print([c for c, i in enumerate(node_anomaly_gator.get_true_anomaly_label_tf_theft()) if i == 1])
# print("The ratio of anomalous nodes to all is", positive_ratio)

# breakpoint()
#--------->  Plot graphs
# for i in [c for c, i in enumerate(node_anomaly_gator.get_true_anomaly_label_tf_theft()) if i == 1]: 
#     if node_anomaly_gator.get_id_theft_prediction()[i] <0.9 :
#         print(f"Showing central node {i}")
#         node_anomaly_gator.plot_graph_IDT(i, plot_only_existing_edges = True)
#         node_anomaly_gator.plot_graph_IDT(i, plot_only_existing_edges = False)
    

#--------->  IMPOSSIBLE EDGES
# sorted_impossible_edges = sorted(node_anomaly_gator.get_number_possible_edges_not_generated())
# plt.bar(*np.unique(sorted_impossible_edges, return_counts = True))
# plt.title("Distribution of Impossible edges")
# plt.xlabel("Number of imposibble edges")
# plt.ylabel("Count in graphs (how many graphs)")
# plt.show()

#---------> anomaly score distribution : 
# plt.hist(node_anomaly_gator.get_id_theft_prediction(), bins = 50)
# plt.title("Node anomaly score distribution")
# plt.show()

plt.figure(figsize=(5, 5))
# scores = node_anomaly_gator.get_id_theft_prediction()
scores = node_anomaly_gator.do_edges_sum_to_degrees()
print(np.mean(scores), np.std(scores))
labels = node_anomaly_gator.get_true_anomaly_label_tf_theft()
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins = 50, color=["#D6A5F1", "#4A21A8"], label=['Normal', 'Anomalous'])
plt.title("Node anomaly score distribution\nalliGATOR")
# plt.title(r'Difference between central node degree and sum of MC probabilities $d_c^0 - W_c$')
plt.xticks(np.arange(min(scores), max(scores), 0.2))
plt.xlabel("Anomaly score")
plt.ylabel("Number of graphs")
plt.yscale('log')
plt.legend()
# plt.savefig("NAD-GATOR-score-dist")
plt.show()

