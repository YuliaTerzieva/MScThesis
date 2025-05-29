from alliGATOR import *


all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)

for n in range(1):

    node_anomaly_gator = alliGATOR(f"./wandb/Synthetic_K10_node/multinomial_diffusion/multistep/2025-05-29_01-17-42", 59, MC = 100, name = f"K{10}DS{8}A{1}", lambda_guidance=4.5, 
                    sample_numbers=275, anomaly_type="node_anomaly", seed=n, tuning = True) 

    precision, recall, auc_precision_recall = node_anomaly_gator.get_PR_AUC(node_anomaly_gator.get_true_anomaly_label_tf_theft(), node_anomaly_gator.get_id_theft_prediction(), title_PR_type="Node Anomaly")

    # precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    
    # all_precision.append(precision_interp)

# all_precision = np.vstack(all_precision)
# mean_precision = all_precision.mean(axis=0)
# std_precision = all_precision.std(axis=0)

# # Plot mean curve with shaded std area
# plt.figure(figsize=(7, 7))
# plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AUC = {auc(interp_recall, mean_precision):.3f})", color = "teal")
# plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.3, color = "teal")

# # Baseline (random)
# labels = node_anomaly_gator.get_true_anomaly_label_tf_theft()
# positive_ratio = labels.count(1)/len(labels)
# plt.hlines(positive_ratio, xmin=0, xmax=1, color='red', label='Baseline')

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Node anomaly\nMean Precision-Recall Curve with Std (alliGATOR)")
# plt.legend()
# plt.show()
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

print([c for c, i in enumerate(node_anomaly_gator.get_true_anomaly_label_tf_theft()) if i == 1])

# breakpoint()
# #--------->  Plot graphs
for i in [135, 136, 207, 217, 233]: 
    node_anomaly_gator.plot_graph_IDT(i, plot_only_existing_edges = True)
    node_anomaly_gator.plot_graph_IDT(i, plot_only_existing_edges = False)
    

#--------->  IMPOSSIBLE EDGES
sorted_impossible_edges = sorted(node_anomaly_gator.get_number_possible_edges_not_generated())
plt.plot(*np.unique(sorted_impossible_edges, return_counts = True))
plt.title("Distribution of Impossible edges")
plt.xlabel("Number of imposibble edges")
plt.ylabel("Count in graphs (how many graphs)")
plt.show()

#---------> anomaly score distribution : 
plt.hist(node_anomaly_gator.get_id_theft_prediction(), bins = 50)
plt.title("Node anomaly score distribution")
plt.show()

scores = node_anomaly_gator.get_id_theft_prediction()
labels = node_anomaly_gator.get_true_anomaly_label_tf_theft()
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
plt.title("Node anomaly score distribution")
plt.legend()
plt.show()
