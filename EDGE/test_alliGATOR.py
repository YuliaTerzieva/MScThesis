from alliGATOR import *



# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-08_19-14-10", 679, MC = 300, name = "ds10_attention8841_cosine", lambda_guidance = 4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention8841_cosine_mc300_guidance45.pkl")

##### so far this is the best one #####
# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_17-27-56", 1069, MC = 1000, name = "ds10_attention21_linear", lambda_guidance = 4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention21_linear_mc1000_guidance45.pkl")

# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_17-27-56", 1069, MC = 1000, name = "ds10_attention21_linear", lambda_guidance = 0, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention21_linear_mc1000_guidance0.pkl")

# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_17-27-56", 1069, MC = 1000, name = "ds10_attention21_linear", lambda_guidance = 1, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention21_linear_mc1000_guidance10.pkl")

# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_19-21-20", 1339, MC = 1000, name = "ds15_attention21_linear", lambda_guidance=4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds15_attention21_linear_mc1000_guidance45.pkl")

# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_21-01-43", 1339, MC = 1000, name = "ds10_attention21_cosine", lambda_guidance=4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention21_cosine_mc1000_guidance45.pkl")

# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-16_19-32-30", 759, MC = 1000, name = "ds10_attention2_linear", lambda_guidance=4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention2_linear_mc1000_guidance45.pkl")


# edge_cls_GATOR = alliGATOR("./wandb/Edge_classification/multinomial_diffusion/multistep/2025-04-24_16-55-08", 619, MC = 1000, name = "edge_cls", lambda_guidance=4.5, 
#                            previously_sampled_model_filename="Alligator_Output/edge_cls_mc1000_guidance45.pkl")

# edge_cls_GATOR.get_PR_AUC(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls(), edge_cls_GATOR.get_edge_cls_anomaly(), title_PR_type = "EDGE classification")

id_theft_GATOR = alliGATOR("./wandb/Id_theft/multinomial_diffusion/multistep/2025-04-27_17-45-43", 619, MC = 500, name = "id_theft", lambda_guidance=4.5, 
                           previously_sampled_model_filename="Alligator_Output/id_theft_mc500_guidance45.pkl")

labels = id_theft_GATOR.get_true_anomaly_label_tf_theft()
print(labels.count(1))

# my_GATOR.plot_active_edges_and_nodes()
# my_GATOR.get_PR_AUC([-lp for lp in my_GATOR.log_graph_probability])
# my_GATOR.get_PR_AUC(my_GATOR.get_anomaly_scores_accounting_for_true_node_degree(), title_PR_type= "Level_of_agreenment")

# my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_min_edge_prob(), title_PR_type = "the min edge probability")
# my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_min_adjusted_edge_prob(), title_PR_type="the min adjusted edge porbability")
# my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_min_adjusted_edge_prob_no_self_loops(), title_PR_type="the min adjusted edge proba without self-loops")

# my_GATOR.get_edge_PR_AUC()
# my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_mean_edge_prob(), title_PR_type=" the mean edge probability")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.get_per_edge_type_probability_list(only_originla_edges=True), "original edges only")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.get_per_edge_type_probability_list_degree_adjusted(only_originla_edges=True), "original edges only")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.get_per_edge_type_probability_list(only_originla_edges=False), "all generated edges")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.get_per_edge_type_probability_list_degree_adjusted_no_self_loops(only_originla_edges=False), "all generated edges")

# --------->  which are the anomalous graphs?

# print([c for c, i in enumerate(edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()) if i == 1])

#--------->  Plot graphs
# for i in [135, 1899, 200, 4, 6, 7]: 
#     edge_cls_GATOR.plot_graph(i, plot_only_existing_edges = True)
    # edge_cls_GATOR.plot_graph(i, plot_only_existing_edges = False)

#--------->  IMPOSSIBLE EDGES
# sorted_impossible_edges = sorted(my_GATOR.get_number_possible_edges_not_generated())
# plt.plot(*np.unique(sorted_impossible_edges, return_counts = True))
# plt.title("Distribution of Impossible edges")
# plt.xlabel("Number of imposibble edges")
# plt.ylabel("Count in graphs (how many graphs)")
# plt.show()
#---------> anomaly score caluclated using the adjusted probabilities : 

# plt.hist(my_GATOR.get_anomaly_scores_accounting_for_true_node_degree(), bins = 50)
# plt.title("Normalised sum of adjusted edge probabilties")
# plt.show()

# scores = my_GATOR.get_anomaly_scores_accounting_for_true_node_degree()
# labels = my_GATOR.get_anomaly_labels_for_original_graphs()
# scores = np.array(scores)
# labels = np.array(labels)
# plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
# plt.title("Normalised sum of adjusted edge probabilties")
# plt.legend()
# plt.show()

# #--

# plt.hist(edge_cls_GATOR.get_edge_cls_anomaly(), bins = 50)
# plt.title("Min adjusted edge probabilties")
# plt.show()

# scores = edge_cls_GATOR.get_edge_cls_anomaly()
# labels = edge_cls_GATOR.get_true_anomaly_labels_for_edge_cls()
# scores = np.array(scores)
# labels = np.array(labels)
# plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
# plt.title("Min adjusted edge probabilties")
# plt.legend()
# plt.show()

# #--

# plt.hist(my_GATOR.get_graph_anomaly_min_edge_prob(), bins = 50)
# plt.title("Min edge probabilties")
# plt.show()

# scores = my_GATOR.get_graph_anomaly_min_edge_prob()
# labels = my_GATOR.get_anomaly_labels_for_original_graphs()
# scores = np.array(scores)
# labels = np.array(labels)
# plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
# plt.title("Min edge probabilities")
# plt.legend()
# plt.show()


#--------->  plot the dists of graph probabilities, caluclated as Ramon's idea
# print(my_GATOR.log_graph_probability)
# print(min(np.exp(my_GATOR.log_graph_probability)))
# print(max(np.exp(my_GATOR.log_graph_probability)))
# print(np.mean(np.exp(my_GATOR.log_graph_probability)))

# sorted_log_graph_probabilities = sorted(np.exp(my_GATOR.log_graph_probability))
# plt.plot(*np.unique(sorted_log_graph_probabilities, return_counts=True))
# plt.title("Distribution of log probabilities of the Graph P(OG | Generated probs)")
# plt.xlabel("Log probability of graph")
# plt.ylabel("Count")
# plt.show()


#--------->  plot the dists of graph probabilities, caluclated as a sum of edge probabilites
# print(my_GATOR.graph_probability_sum_existing_edges)
# print(min(np.exp(my_GATOR.graph_probability_sum_existing_edges)))
# print(max(np.exp(my_GATOR.graph_probability_sum_existing_edges)))
# print(np.mean(np.exp(my_GATOR.graph_probability_sum_existing_edges)))

# sorted_log_graph_probabilities = sorted(np.exp(my_GATOR.graph_probability_sum_existing_edges))
# plt.plot(*np.unique(sorted_log_graph_probabilities, return_counts=True))
# plt.title("Distribution of Graph probability, claculated as a sum of the probabilities of origianl edges ")
# plt.xlabel("Probability of graph")
# plt.ylabel("Count")
# plt.show()