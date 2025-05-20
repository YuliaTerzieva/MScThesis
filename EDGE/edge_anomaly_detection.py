from alliGATOR import *

all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)

for n in range(10):

    edge_cls_GATOR = alliGATOR("./wandb/Edge_classification_K15/multinomial_diffusion/multistep/K15DS10A1", 559, MC = 100, name = "edge_cls", lambda_guidance=0.5, 
                               sample_numbers=1000, seed=n)

    precision, recall, auc_precision_recall = edge_cls_GATOR.get_PR_AUC(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls(), edge_cls_GATOR.get_edge_cls_anomaly(), title_PR_type = "EDGE classification")

    precision_interp = np.interp(interp_recall, recall[::-1], precision[::-1], left=1.0)
    
    all_precision.append(precision_interp)

all_precision = np.vstack(all_precision)
mean_precision = all_precision.mean(axis=0)
std_precision = all_precision.std(axis=0)

# Plot mean curve with shaded std area
plt.figure(figsize=(7, 7))
plt.plot(interp_recall, mean_precision, label=f"Mean PR curve (AUC = {auc(interp_recall, mean_precision):.3f})", color = "teal")
plt.fill_between(interp_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.3, color = "teal")

# Baseline (random)
labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
positive_ratio = labels.count(1)/len(labels)
plt.hlines(positive_ratio, xmin=0, xmax=1, color='red', label='Baseline')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Edge anomaly\nMean Precision-Recall Curve with Std (alliGATOR)")
plt.legend()
plt.show()


# edge_cls_GATOR.plot_active_edges_and_nodes()
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=True), "original edges only")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=True, node_degree_adjusted=True), "original edges only\nadjusted edge probs")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=False), "all generated edges")
# edge_cls_GATOR.plot_edge_distribution_violin_boxplots(edge_cls_GATOR.get_per_edge_type_probability_list(only_originla_edges=False, node_degree_adjusted=True), "all generated edges\nadjusted edge probs")

# --------->  which are the anomalous graphs?

# print([c for c, i in enumerate(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()) if i == 1])

# #--------->  Plot graphs
# for i in [135, 1899, 200, 4, 6, 7]: 
#     edge_cls_GATOR.plot_graph_edge_cls(i, plot_only_existing_edges = True)
#     edge_cls_GATOR.plot_graph_edge_cls(i, plot_only_existing_edges = False)

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

# scores = edge_cls_GATOR.get_edge_cls_anomaly()
# labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
# scores = np.array(scores)
# labels = np.array(labels)
# plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
# plt.title("Edge classification predicted dist")
# plt.legend()
# plt.show()

