from alliGATOR import *

# id_theft_GATOR = alliGATOR("./wandb/Id_theft/multinomial_diffusion/multistep/2025-04-27_19-36-11", 319, MC = 100, name = "id_theft", lambda_guidance=4.5, 
#                            sample_numbers=1000)
id_theft_GATOR = alliGATOR("./wandb/Id_theft_2/multinomial_diffusion/multistep/2025-04-27_22-09-16", 919, MC = 500, name = "id_theft_2", lambda_guidance=4.5, 
                           sample_numbers=1000, previously_sampled_model_filename="Alligator_Output/id_theft_2_mc500_guidance45.pkl")

# id_theft_GATOR.get_PR_AUC(id_theft_GATOR.get_true_anomaly_label_tf_theft(), id_theft_GATOR.get_id_theft_prediction(), title_PR_type="Node Anomaly")
# for this without noamlization we have : 0.50693, with 2m we have 0.39822; with 2m**2 we have 0.52919 and with 4m**2 we have 0.51455


id_theft_GATOR.plot_active_edges_and_nodes()
# id_theft_GATOR.plot_edge_distribution_violin_boxplots(id_theft_GATOR.get_per_edge_type_probability_list(only_originla_edges=True), "original edges only")
# id_theft_GATOR.plot_edge_distribution_violin_boxplots(id_theft_GATOR.get_per_edge_type_probability_list(only_originla_edges=True, node_degree_adjusted=True), "original edges only")
# id_theft_GATOR.plot_edge_distribution_violin_boxplots(id_theft_GATOR.get_per_edge_type_probability_list(only_originla_edges=False), "all generated edges")
# id_theft_GATOR.plot_edge_distribution_violin_boxplots(id_theft_GATOR.get_per_edge_type_probability_list(only_originla_edges=False, node_degree_adjusted=True), "all generated edges")

# --------->  which are the anomalous graphs?

# print([c for c, i in enumerate(id_theft_GATOR.get_true_anomaly_label_tf_theft()) if i == 1])

# #--------->  Plot graphs
# for i in [61, 229, 3]: 
#     id_theft_GATOR.plot_graph_IDT(i, plot_only_existing_edges = False)
#     id_theft_GATOR.plot_graph_IDT(i, plot_only_existing_edges = True)


#--------->  IMPOSSIBLE EDGES
# sorted_impossible_edges = sorted(id_theft_GATOR.get_number_possible_edges_not_generated())
# plt.plot(*np.unique(sorted_impossible_edges, return_counts = True))
# plt.title("Distribution of Impossible edges")
# plt.xlabel("Number of imposibble edges")
# plt.ylabel("Count in graphs (how many graphs)")
# plt.show()

#---------> anomaly score distribution : 
# plt.hist(id_theft_GATOR.get_id_theft_prediction(), bins = 50)
# plt.title("Identity theft")
# plt.show()

# scores = id_theft_GATOR.get_id_theft_prediction()
# labels = id_theft_GATOR.get_true_anomaly_label_tf_theft()
# scores = np.array(scores)
# labels = np.array(labels)
# plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
# plt.title("Identity Theft")
# plt.legend()
# plt.show()
