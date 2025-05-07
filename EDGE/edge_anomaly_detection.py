from alliGATOR import *


edge_cls_GATOR = alliGATOR("./wandb/Edge_classification/multinomial_diffusion/multistep/2025-04-27_18-51-17", 619, MC = 1000, name = "edge_cls", lambda_guidance=4.5, 
                           previously_sampled_model_filename="Alligator_Output/edge_cls_mc1000_guidance45.pkl", sample_numbers=2186)

edge_cls_GATOR.get_PR_AUC(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls(), edge_cls_GATOR.get_edge_cls_anomaly(), title_PR_type = "EDGE classification")
# for this without noamlization we have : 0.87929, with 2m we have 0.04272; with 2m**2 we have 0.8845 and with 4m**2 we have 0.8822


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

plt.hist(edge_cls_GATOR.get_edge_cls_anomaly(), bins = 50)
plt.title("Edge classification predicted dist")
plt.show()

scores = edge_cls_GATOR.get_edge_cls_anomaly()
labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
plt.title("Edge classification predicted dist")
plt.legend()
plt.show()

