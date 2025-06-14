from alliGATOR import *

all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)
all_avg_precision = []

for n in range(1):

    edge_cls_GATOR = alliGATOR(f"./wandb/Synthetic_K7_edge/multinomial_diffusion/multistep/DS5A331", 212, MC = 100, name = "edge_cls-node-norm", lambda_guidance=4.5, 
                               sample_numbers=1894, previously_sampled_model_filename="./Alligator_Output_edge_anomaly/edge_cls-node-norm_mc100_guidance45.pkl", tuning=False, anomaly_type="edge_anomaly", seed=n)

#     precision, recall, auc_precision_recall, average_precision = edge_cls_GATOR.get_PR_AUC(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls(), edge_cls_GATOR.get_edge_cls_anomaly(), title_PR_type = "EDGE classification")

#     precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    
#     all_precision.append(precision_interp)
#     all_avg_precision.append(average_precision)

# all_precision = np.vstack(all_precision)
# mean_precision = all_precision.mean(axis=0)
# std_precision = all_precision.std(axis=0)

# # Plot mean curve with shaded std area
# plt.figure(figsize=(5, 5))
# plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AP = {np.mean(all_avg_precision):.5f})", color = "#4A21A8")
# plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.5, color = "#7DC3F6")

# # Baseline (random)
# labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
# positive_ratio = labels.count(1)/len(labels)
# plt.hlines(positive_ratio, xmin=0, xmax=1, color="#F1993A", label='Baseline')

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Edge anomaly on Synthetic dataset \nMean Precision-Recall Curve with Std (alliGATOR)")
# plt.legend()
# plt.savefig("?")


# edge_cls_GATOR.plot_active_edges_and_nodes()
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=True), "original edges only")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=True, node_degree_adjusted=True), "original edges only\nadjusted edge probs")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=False), "all generated edges")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=False, node_degree_adjusted=True), "all generated edges\nadjusted edge probs")

# --------->  which are the anomalous graphs?

# print([c for c, i in enumerate(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()) if i == 1])

# #--------->  Plot graphs
for i in [c for c, i in enumerate(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()) if i == 0]: 
    if edge_cls_GATOR.get_edge_cls_anomaly()[i] > 1.3 :
        edge_cls_GATOR.plot_graph_edge_cls(i, plot_only_existing_edges = True)
    # edge_cls_GATOR.plot_graph_edge_cls(i, plot_only_existing_edges = False)

#--------->  IMPOSSIBLE EDGES
# sorted_impossible_edges = sorted(edge_cls_GATOR.get_number_possible_edges_not_generated())
# plt.plot(*np.unique(sorted_impossible_edges, return_counts = True))
# plt.title("Distribution of Impossible edges")
# plt.xlabel("Number of imposibble edges")
# plt.ylabel("Count in graphs (how many graphs)")
# plt.show()
#---------> anomaly score distribution : 

# plt.hist(edge_cls_GATOR.get_edge_cls_anomaly(), bins = 50)
# plt.title("Edge classification predicted dist")
# plt.show()

plt.figure(figsize=(5, 5))
scores = edge_cls_GATOR.get_edge_cls_anomaly()
labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins = 50, color=["#D6A5F1", "#4A21A8"], label=['Normal', 'Anomalous'])
plt.title("Edge anomaly score distribution\nalliGATOR")
plt.xticks(np.arange(min(scores), max(scores), 0.2))
plt.xlabel("Anomaly score")
plt.ylabel("Number of graphs")
plt.yscale('log')
plt.legend()
plt.savefig("dist with edge norm")
plt.show()

