from alliGATOR import *



# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-08_19-14-10", 679, MC = 300, name = "ds10_attention8841_cosine", lambda_guidance = 4.5, 
#                      previously_sampled_model_filename="Alligator_Output/ds10_attention8841_cosine_mc300_guidance45.pkl")

my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-13_17-27-56", 1069, MC = 1000, name = "ds10_attention21_linear", lambda_guidance = 4.5, 
                     previously_sampled_model_filename="Alligator_Output/ds10_attention21_linear_mc1000_guidance45.pkl")

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


# my_GATOR.plot_active_edges_and_nodes()
# my_GATOR.get_PR_AUC([-lp for lp in my_GATOR.log_graph_probability])
my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_min_edge_prob(), title_PR_type = "the min edge probability")
# my_GATOR.get_edge_PR_AUC()
# my_GATOR.get_PR_AUC(my_GATOR.get_graph_anomaly_mean_edge_prob(), title_PR_type=" the mean edge probability")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.edge_type_probability_original_edges, "original edges only")
# my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.edge_type_probability_across_generated, "all generated edges")

#--------->  which are the anomalous graphs?
# for count, graph in enumerate(my_GATOR.original_graphs):
#     if graph.edge_anomalous.any() :
#         print(count)

# for i in [2, 3, 13, 21, 800]: 
#     my_GATOR.plot_graph(i, plot_only_existing_edges = True)
#     my_GATOR.plot_graph(i, plot_only_existing_edges = False)

#--------->  IMPOSSIBLE EDGES
# sorted_impossible_edges = sorted(my_GATOR.number_possible_edges_not_generated)
# plt.plot(*np.unique(sorted_impossible_edges, return_counts = True))
# plt.title("Distribution of Impossible edges")
# plt.xlabel("Number of imposibble edges")
# plt.ylabel("Count in graphs (how many graphs)")
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