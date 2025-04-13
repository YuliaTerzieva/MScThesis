from alliGATOR import *


# my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-08_19-14-10", 679, MC = 300, lambda_guidance = 4.5)



my_GATOR = alliGATOR("./wandb/RelationalDataset_with_anomaly/multinomial_diffusion/multistep/2025-04-08_19-14-10", 679, MC = 300, lambda_guidance = 4.5, 
                     previously_sampled_model_filename = "Alligator_Output/sampled_mc300_guidance45.pkl", node_color_mapping={0: 'blue', 1: 'orange', 2: 'grey'})

my_GATOR.get_PR_AUC()
my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.edge_type_probability_original_edges, "original edges only")
my_GATOR.plot_edge_distribution_violin_boxplots(my_GATOR.edge_type_probability_across_generated, "all generated edges")

#--------->  which are the anomalous graphs?
# for count, graph in enumerate(my_GATOR.original_graphs):
#     if graph.edge_anomalous.any() :
#         print(count)

# for i in [2, 13, 21]: 
#     my_GATOR.plot_graph(i, plot_only_existing_edges = True)
#     my_GATOR.plot_graph(i, plot_only_existing_edges = False)



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