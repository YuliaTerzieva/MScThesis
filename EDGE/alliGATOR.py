import torch
import torch_geometric as pyg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, Counter
import pickle
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
import time

# ---- EDGE functions ----
from datasets.data import get_data
from model import get_model


class alliGATOR(object):

    def __init__(self, saved_model, checkpoint_nb, MC, name, lambda_guidance = 4.5, previously_sampled_model_filename = None, sample_numbers = 1, anomaly_type = "", node_color_mapping = {0: 'blue', 1: 'orange', 2: 'grey'}, seed = 42):
        
        wandb_model_log = saved_model
        path = wandb_model_log+f"/check/checkpoint_{checkpoint_nb}.pt"
        path_args = wandb_model_log+ "/args.pickle"
        with open(path_args, 'rb') as f:
            args = pickle.load(f)
        args.device = 'cpu'
        _, _, _, _, _, _, _, _, initial_graph_sampler, _, _, _ = get_data(args)
        self.number_diffusion_steps = args.diffusion_steps

        self.sample_numbers = np.arange(sample_numbers).tolist() # for the classification we have 2186, for the id theft we have 1000
        self.Monte_Carlo = MC

        self.node_color = node_color_mapping

        self.model = get_model(args, initial_graph_sampler=initial_graph_sampler)
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])

        if torch.cuda.is_available():
            self.model = self.model.to(args.device)
        self.model.eval()

        if previously_sampled_model_filename == None:
            # 0 only conditioned, >0 subtracks the unconditioned, actively reducing the probability of generating samples that ignore the conditioning
            self.original_graphs, self.generated_graphs, self.per_graph_edge_list_counter, self.active_edges = self.model.sample_and_MC(self.sample_numbers, lambda_guidance, self.Monte_Carlo, seed = seed) 
            # with open(f"Alligator_Output_{anomaly_type}/{name}_mc{self.Monte_Carlo}_guidance{int(lambda_guidance*10)}.pkl", "wb") as f:
            #     pickle.dump([self.original_graphs, self.generated_graphs, self.per_graph_edge_list_counter, self.active_edges], f)
        else :
            with open(previously_sampled_model_filename, 'rb') as f:
                self.original_graphs, self.generated_graphs, self.per_graph_edge_list_counter, self.active_edges = pickle.load(f)

        self.original_graphs_edges = self.get_original_graphs_edges() # this is dictionary
    
#--------------- General functions ---------------

    def get_original_graphs_edges(self) -> dict[list[(int, int)]] :
        """This function returns a list of edges (node tuples) for each graph from the original/testing graphs"""

        original_graphs_edges = defaultdict(list)

        for g_nb, graph in enumerate(self.original_graphs):
            original_edges = zip(*graph.edge_index.tolist())
            original_graphs_edges[g_nb] = sorted({tuple(sorted((u, v))) for u, v in original_edges})

        return original_graphs_edges
    
    def get_number_possible_edges_not_generated(self)->list:
        # for each graph how many edges are "impossible", i.e. not generated at all and not in the original graph
        number_impossible_edges = np.zeros(len(self.original_graphs))

        for g_nb, graph in enumerate(self.original_graphs):

            # the possible number of edges in undirected graph with no self-edges is n(n-1)/2
            impossible_edges = ((graph.num_nodes * (graph.num_nodes - 1))/2) - len(self.per_graph_edge_list_counter[g_nb].keys())
            number_impossible_edges[g_nb] = impossible_edges

        return number_impossible_edges.tolist()

    def get_per_edge_type_probability_list(self, only_originla_edges = False, node_degree_adjusted = False) -> dict[list[int]]:
        """
        Creating a list of probabilities for each edge type (e.g. Blue-Blue)
        This list is generated from all the edges generated during the inference for all graphs. 
        If only_originla_edges is true, then only edges that were in the original graph are selected. 
        -------
        edge_prob_stats: dict
            Format: {
                (node_type, node_type): [prob1, prob2, ...]
            }
        """

        edge_prob_stats = defaultdict(list)

        # this is for the node degree adjusted scenario
        per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()

        for g_idx, graph in enumerate(self.original_graphs):
            edge_counts = self.per_graph_edge_list_counter[g_idx]

            # this is for the node degree adjusted scenario
            node_degree_counter = per_graph_node_degree_counter[g_idx]
            m = len(self.original_graphs_edges[g_idx])


            for (u, v), count in edge_counts.items():
                if not only_originla_edges or ((u, v) in self.original_graphs_edges[g_idx]):
                    # for each generated edge for that graph save its probability 
                    node_type_u = int(graph.node_attr[u])
                    node_type_v = int(graph.node_attr[v])
                    edge_type = tuple(sorted((node_type_u, node_type_v)))
                     
                    if node_degree_adjusted :
                        probability = count / self.Monte_Carlo - ((node_degree_counter[u] * node_degree_counter[v]) / (2*m))
                    else:
                        probability = count / self.Monte_Carlo

                    edge_prob_stats[edge_type].append(probability)

        return edge_prob_stats
        
#--------------- Anomaly detection functions ---------------

    # def get_graph_anomaly_min_edge_prob(self) -> np.array:
    #     """
    #     The anomaly score of a graph is 1 - min(edge probabilities of G_c)
    #     """
    #     anomaly_score_per_graph = np.zeros(len(self.sample_numbers))

    #     for g_index in range(len(self.sample_numbers)):

    #         generated_edges_prob = self.per_graph_edge_list_counter[g_index]
    #         anomaly_score_per_graph[g_index] = 1 - min([generated_edges_prob[edge] / self.Monte_Carlo  for edge in self.original_graphs_edges[g_index] if edge in generated_edges_prob.keys()])

    #     return anomaly_score_per_graph
    
    # def get_graph_anomaly_min_adjusted_edge_prob(self) -> np.array:
    #     """
    #     The anomaly score of a graph is 1 - min(edge probabilities of G_c adjusted for node degree per node)

    #     this is following the formula by Ioana, but i am not sure it is correctly implemented
    #     """
    #     anomaly_score_per_graph = np.zeros(len(self.sample_numbers))
    #     per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()
        
    #     for g_index in range(len(self.sample_numbers)):
    #         generated_edges_prob = self.per_graph_edge_list_counter[g_index]
    #         m = len(self.original_graphs_edges[g_index])
    #         node_degree_counter = per_graph_node_degree_counter[g_index]
    #         anomaly_score_per_graph[g_index] = 1 - min([generated_edges_prob[edge] / self.Monte_Carlo 
    #                                                     - ((node_degree_counter[edge[0]] * node_degree_counter[edge[1]]) / (2*m)) 
    #                                                     for edge in self.original_graphs_edges[g_index] if edge in generated_edges_prob.keys()])

    #     return anomaly_score_per_graph

    # def get_anomaly_scores_accounting_for_true_node_degree(self) -> np.array:

        anomaly_scores = np.zeros(len(self.per_graph_edge_list_counter))
        per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()

        for graph_nb, (original_graph_edges, gen_edge_list_counter, node_degree_counter) in enumerate(zip(self.original_graphs_edges.values(), self.per_graph_edge_list_counter, per_graph_node_degree_counter)):

            # breakpoint()

            m = len(original_graph_edges)

            # Wij = (gen_edge_list_counter[edge] / self.Monte_Carlo)
            # kikj / 2m = (node_degree_counter[edge[0]] * node_degree_counter[edge[1]]) / (2*m)
            numerator = sum([((gen_edge_list_counter[edge] / self.Monte_Carlo) - ((node_degree_counter[edge[0]] * node_degree_counter[edge[1]]) / (2*m))) 
                             for edge in original_graph_edges])
            # print([((gen_edge_list_counter[edge] / self.Monte_Carlo) ) 
            #                  for edge in original_graph_edges])
            # print([(((node_degree_counter[edge[0]] * node_degree_counter[edge[1]]) / (2*m))) 
            #                  for edge in original_graph_edges])
            # print([((gen_edge_list_counter[edge] / self.Monte_Carlo) - ((node_degree_counter[edge[0]] * node_degree_counter[edge[1]]) / (2*m))) 
            #                  for edge in original_graph_edges])
            denominator = m - (1 / (2 * m)) * sum([node_degree_counter[edge[0]] * node_degree_counter[edge[1]] for edge in original_graph_edges])
            level_of_agreenment = numerator / denominator

            anomaly_scores[graph_nb] =  1 - level_of_agreenment

        return anomaly_scores

    def get_per_graph_node_degree_counter(self) -> list[Counter]:

        per_graph_node_degree_counter = []

        for edge_tuple_list in self.original_graphs_edges.values():
            counter = Counter()
            for edge in edge_tuple_list :
                counter.update([edge[0], edge[1]])
            per_graph_node_degree_counter.append(counter)
        
        return per_graph_node_degree_counter

#----- >  EDGE CLS 
    def get_true_anomaly_labels_for_edge_cls(self)->list:

        labels = np.zeros(len(self.original_graphs))

        for g_index in range(len(self.sample_numbers)):
            central_edge_index = np.where(self.original_graphs[g_index]['central_edge'] == 1)[0]
            if self.original_graphs[g_index].anomalous[central_edge_index].any() :
                labels[g_index] = 1

        return labels.tolist()

    def get_edge_cls_anomaly(self) -> np.array :

        anomaly_score_per_edge_sub_graph = np.zeros(len(self.sample_numbers))
        central_edge_list = self.get_central_edge_per_graph()
        per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()
        
        for g_index in range(len(self.sample_numbers)):
            central_edge = central_edge_list[g_index]
            m = len(self.original_graphs_edges[g_index])
            k = per_graph_node_degree_counter[g_index]

            # to_norm = 1 - (1/(2*m**2)) * np.sum([(k[edge[0]] * k[edge[1]])
            #                                     for edge in self.original_graphs_edges[g_index]])
        
            generated_prob_central_edge = self.per_graph_edge_list_counter[g_index][central_edge] / self.Monte_Carlo #- ((k[central_edge[0]] * k[central_edge[1]]) / (2*m))) / to_norm

            anomaly_score_per_edge_sub_graph[g_index] = 1 - generated_prob_central_edge

        return anomaly_score_per_edge_sub_graph

    def get_central_edge_per_graph(self) -> list[tuple[int, int]] :

        central_edge_list = []

        for g_index in range(len(self.sample_numbers)):

            central_edge_index = (np.where(self.original_graphs[g_index]['central_edge'] == 1)[0][0], np.where(self.original_graphs[g_index]['central_edge'] == 1)[0][1])
            central_edge = tuple(sorted((self.original_graphs[g_index].edge_index[0][central_edge_index[0]].item(), self.original_graphs[g_index].edge_index[0][central_edge_index[1]].item())))
            
            central_edge_list.append(central_edge)


        return central_edge_list
    
#----- > Id Theft
    def get_true_anomaly_label_tf_theft(self) -> list:
        labels = np.zeros(len(self.original_graphs))

        for g_index in range(len(self.sample_numbers)):
            central_node_index = np.where(self.original_graphs[g_index].central_node == 1)[0]
            if self.original_graphs[g_index].anomalous[central_node_index].any() :
                # breakpoint() 
                labels[g_index] = 1

        return labels.tolist()
    
    def get_id_theft_prediction(self) -> np.array :

        anomaly_score_per_edge_sub_graph = np.zeros(len(self.sample_numbers))

        for g_index in range(len(self.sample_numbers)):
            central_node_index = np.where(self.original_graphs[g_index].central_node == 1)[0]
            central_node_edges = [edge for edge in self.original_graphs_edges[g_index] if central_node_index in edge]

            gen_edges = self.per_graph_edge_list_counter[g_index]            
            score = np.mean([gen_edges[edge]/self.Monte_Carlo for edge in central_node_edges])
            
            node_anomaly = 1 - score
            anomaly_score_per_edge_sub_graph[g_index] = node_anomaly

        return anomaly_score_per_edge_sub_graph

#----- > Cora dataset
    def get_true_anomaly_label_core(self) -> list:
        labels = np.zeros(len(self.original_graphs))

        for g_index in range(len(self.sample_numbers)):
            central_node_index = np.where(self.original_graphs[g_index].central_node == 1)[0]
            if self.original_graphs[g_index].ano_label[central_node_index].any() :
                labels[g_index] = 1

        return labels.tolist()

    def get_cora_prediction(self) -> np.array :

        # breakpoint()
        anomaly_score_per_edge_sub_graph = np.zeros(len(self.sample_numbers))

        for g_index in range(len(self.sample_numbers)):
            central_node_index = np.where(self.original_graphs[g_index].central_node == 1)[0]
            central_node_edges = [edge for edge in self.original_graphs_edges[g_index] if central_node_index in edge]

            gen_edges = self.per_graph_edge_list_counter[g_index]            
            score = np.mean([gen_edges[edge]/self.Monte_Carlo for edge in central_node_edges]) # TODO : here i need to add 0 for every edge that was not generated!
            
            node_anomaly = 1 - score
            anomaly_score_per_edge_sub_graph[g_index] = node_anomaly

        return anomaly_score_per_edge_sub_graph
#----- > PR AUC
    def get_PR_AUC(self, true_labels, predicted_labels, title_PR_type) -> float:

        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(true_labels, predicted_labels)
        # breakpoint()
        # print("Tresholds : ", thresholds)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        print('\033[1;36m'+f"The ACU using {title_PR_type} of a graph is {round(auc_precision_recall, 5)}\n"+'\033[0;36m')

        # plt.figure(figsize=(7, 7))
        # plt.plot(recall, precision, label = f"Alligator with AUC = {round(auc_precision_recall, 3)}", color='green')
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title(f"Precision - Recall curve")# with AUC = {round(auc_precision_recall, 3)} using {title_PR_type}")

        # # now the "baseline", nice explanation here : https://stats.stackexchange.com/questions/251175/what-is-baseline-in-precision-recall-curve
        # plt.hlines(true_labels.count(1)/len(true_labels), xmin=0, xmax=1, label = "Baseline curve", color='red')
        # plt.legend()
        # plt.show()

        return precision, recall, auc_precision_recall

                                                 
#--------------- Plots ---------------

    def plot_active_edges_and_nodes(self):

        timesteps = np.arange(self.number_diffusion_steps-1, -1, -1)
        vals = [self.active_edges[t] for t in timesteps]
        val1s, val2s, vals3s = zip(*vals)

        # Plot in one go without separate variable assignment
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, val1s, label='Number of active edges')
        plt.plot(timesteps, val2s, label='Number of added edges')
        plt.title('Values over Timesteps')
        plt.xlabel('Timestep (t)')
        plt.ylabel('Values')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, vals3s, label='Number of active nodes')
        plt.hlines(self.generated_graphs.num_nodes, timesteps[0], timesteps[-1], label='Total number of nodes')
        plt.title('Values over Timesteps')
        plt.xlabel('Timestep (t)')
        plt.ylabel('Values')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return

    def plot_edge_distribution_violin_boxplots(self, probabilities_per_edge_type, print_string_type_data) -> None:
        """
        Plots a box plot and a violin plot for the distribution of edge type probabilities
        for all edge types: blue-blue, blue-orange, blue-grey, orange-orange, orange-grey, and grey-grey.
        Each plot is colored based on the node colors. For mixed edge types, the colors are blended.
        
        Parameters:
        -----------
        probabilities_per_edge_type : dict
            Dictionary of edge-type probability distributions 
            Format: {
                (node_type_1, node_typ1_2): [prob1, prob2, ...]
            }
        print_string_type_data : str
            Used in the titles of the plots, to distinguish what type of data (i.e. probabilities_per_edge_type) the function recieved
            Can be "original edges only" or "all generated edges"
        """

        edge_type_order = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        labels = [f"{self.node_color[a]}-{self.node_color[b]}" for a, b in edge_type_order]
        
        # Create an edge color mapping. For mixed edges, blend the two colors.
        edge_color_map = {}
        for etype in edge_type_order:
            if etype[0] == etype[1]:
                edge_color_map[etype] = self.node_color[etype[0]]
            else:
                rgb1 = np.array(mcolors.to_rgb(self.node_color[etype[0]]))
                rgb2 = np.array(mcolors.to_rgb(self.node_color[etype[1]]))
                blended = (rgb1 + rgb2) / 2
                edge_color_map[etype] = mcolors.to_hex(blended)
        
        # for each possible relation, get the probability distribution or an empty array
        data = [probabilities_per_edge_type.get(etype, [0]) for etype in edge_type_order]
        

        for relation_number, relation_probabilities in enumerate(data):
            sorted_relation_probabilities = sorted(relation_probabilities)
            plt.plot(*np.unique(sorted_relation_probabilities, return_counts=True), label = f"Relation {edge_type_order[relation_number]}")
        plt.legend()
        plt.title(f"Probability distribuiton per edge type from {print_string_type_data} data")    
        plt.show()


        # Create a figure with two subplots: one for the box plot and one for the violin plot.
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # -- Box Plot --
        bp = axes[0].boxplot(data, patch_artist=True, tick_labels=labels)
        # Color each box with the corresponding edge type color.
        for patch, etype in zip(bp['boxes'], edge_type_order):
            patch.set_facecolor(edge_color_map[etype])
            patch.set_alpha(0.7)
        axes[0].set_title(f"Box Plot of Edge Type Probabilities from {print_string_type_data} data")
        axes[0].set_xlabel("Edge Types")
        axes[0].set_ylabel("Probability")
        
        # # -- Violin Plot --
        vp = axes[1].violinplot(data, showmeans=True, showmedians=True, showextrema=True)
        # Color each violin body.
        for i, body in enumerate(vp['bodies']):
            etype = edge_type_order[i]
            body.set_facecolor(edge_color_map[etype])
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        
        axes[1].set_title(f"Violin Plot of Edge Type Probabilities from {print_string_type_data} data")
        axes[1].set_xlabel("Edge Types")
        axes[1].set_ylabel("Probability")
        axes[1].set_xticks(np.arange(1, len(labels) + 1))
        axes[1].set_xticklabels(labels)
        
        plt.tight_layout()
        plt.show()

    def plot_graph_edge_cls(self, graph_id, plot_only_existing_edges):

        original_graph = self.original_graphs[graph_id] # this is pyg data object
        generated_graph = self.generated_graphs[graph_id]
        generated_edges = self.per_graph_edge_list_counter[graph_id]
        central_edge = self.get_central_edge_per_graph()[graph_id]

        # breakpoint()

        # from pyg to networkx original graph : 
        original_node_colors = [self.node_color[node_class.item()] for node_class in original_graph.node_attr]
        original_anomalous_edges = [(u.item(), v.item()) for (u, v), is_anom in zip(zip(original_graph.edge_index[0], original_graph.edge_index[1]), original_graph.anomalous) if is_anom]
        original_graph_nx = pyg.utils.to_networkx(original_graph, to_undirected=True)

        # generated edges/graph
        generated_node_colors = [self.node_color[node_class.item()] for node_class in generated_graph.node_attr]
        generated_graph_nx = pyg.utils.to_networkx(generated_graph, to_undirected=True)
        generated_graph_nx.clear_edges()
        # ------  Probabilities ------
        # edges_with_weights = [(u, v, {'probability': count / self.Monte_Carlo}) for (u, v), count in generated_edges.items() if not plot_only_existing_edges or (u, v) in self.original_graphs_edges[graph_id]]
        # ------  END of Probabilities ------

        # ------  Adjusted probabilities ------
        per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()
        m = len(self.original_graphs_edges[graph_id])
        k = per_graph_node_degree_counter[graph_id]
        edges_with_weights = [(u, v, {'probability': count / self.Monte_Carlo, 
                                      'adjusted_prob': count / self.Monte_Carlo - (k[u] * k[v]) / (2*m)}) 
                               for (u, v), count in generated_edges.items() if not plot_only_existing_edges or (u, v) in self.original_graphs_edges[graph_id]]
        # print(edges_with_weights)
        # print(f"M is {m}, and the ks are {per_graph_node_degree_counter[graph_id]}")
        # ------  END of Adjusted probabilities ------


        generated_graph_nx.add_edges_from(edges_with_weights)
        edge_labels = {(u, v): f"{data['probability']:.2f} -> {data['adjusted_prob']:.2f}" for u, v, data in generated_graph_nx.edges(data=True)}

        fig, axes = plt.subplots(1, 2)

        # plot the original graph and add red to the anomalous edges
        pos = nx.circular_layout(original_graph_nx)
        OG_edge_colors = ["red" if e in original_anomalous_edges else "black" for e in original_graph_nx.edges()]
        nx.draw(original_graph_nx, pos, ax=axes[0], with_labels=True, node_color=original_node_colors, edge_color = OG_edge_colors)

        # plot the generated graph 
        pos = nx.circular_layout(generated_graph_nx)
        nx.draw(generated_graph_nx, pos, ax=axes[1], with_labels=True, node_color=generated_node_colors)
        nx.draw_networkx_edge_labels(generated_graph_nx, pos, ax=axes[1], edge_labels=edge_labels, font_color='gray')
        
        axes[0].set_title(f"Original graph with central edge {central_edge}")
        axes[1].set_title(f"Edge probability over {self.Monte_Carlo} generated graphs")
        plt.tight_layout()
        plt.show() 

    def plot_graph_IDT(self, graph_id, plot_only_existing_edges):

        original_graph = self.original_graphs[graph_id] # this is pyg data object
        generated_graph = self.generated_graphs[graph_id]
        generated_edges = self.per_graph_edge_list_counter[graph_id]


        original_node_colors = [self.node_color[node_class.item()] for node_class in original_graph.node_attr]
        original_graph_nx = pyg.utils.to_networkx(original_graph, to_undirected=True)

        # generated edges/graph
        generated_node_colors = [self.node_color[node_class.item()] for node_class in generated_graph.node_attr]
        generated_graph_nx = pyg.utils.to_networkx(generated_graph, to_undirected=True)
        generated_graph_nx.clear_edges()
        # ------  Probabilities ------
        # edges_with_weights = [(u, v, {'probability': count / self.Monte_Carlo}) for (u, v), count in generated_edges.items() if not plot_only_existing_edges or (u, v) in self.original_graphs_edges[graph_id]]
        # ------  END of Probabilities ------

        # ------  Adjusted probabilities ------
        per_graph_node_degree_counter = self.get_per_graph_node_degree_counter()
        m = len(self.original_graphs_edges[graph_id])
        k = per_graph_node_degree_counter[graph_id]
        edges_with_weights = [(u, v, {'probability': count / self.Monte_Carlo, 
                                      'adjusted_prob': count / self.Monte_Carlo - (k[u] * k[v]) / (2*m**2)}) 
                               for (u, v), count in generated_edges.items() if not plot_only_existing_edges or (u, v) in self.original_graphs_edges[graph_id]]
        # print(edges_with_weights)
        # print(f"M is {m}, and the ks are {per_graph_node_degree_counter[graph_id]}")
        # ------  END of Adjusted probabilities ------


        generated_graph_nx.add_edges_from(edges_with_weights)
        edge_labels = {(u, v): f"{data['probability']:.2f} -> {data['adjusted_prob']:.2f}" for u, v, data in generated_graph_nx.edges(data=True)}

        fig, axes = plt.subplots(1, 2)

        # plot the original graph and add red to the anomalous edges
        pos = nx.circular_layout(original_graph_nx)
        nx.draw(original_graph_nx, pos, ax=axes[0], with_labels=True, node_color=original_node_colors)

        # plot the generated graph 
        pos = nx.circular_layout(generated_graph_nx)
        nx.draw(generated_graph_nx, pos, ax=axes[1], with_labels=True, node_color=generated_node_colors)
        nx.draw_networkx_edge_labels(generated_graph_nx, pos, ax=axes[1], edge_labels=edge_labels, font_color='gray')
        

        central_node_index = np.where(self.original_graphs[graph_id].central_node == 1)[0]
        axes[0].set_title(f"Original graph with central node {central_node_index}")
        axes[1].set_title(f"Edge probability over {self.Monte_Carlo} generated graphs")
        plt.tight_layout()
        plt.show() 