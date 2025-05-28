from alliGATOR import *

all_precision = []
n_interp_points = 500
interp_recall = np.linspace(0, 1, n_interp_points)

for n in range(1):

    node_anomaly_gator = alliGATOR("./wandb/Cora_inductive_PCA30_node/multinomial_diffusion/multistep/2025-05-28_16-34-42", 19, MC = 100, name = f"Cora_inductive_output", anomaly_type="cora", lambda_guidance=4.5, 
                    sample_numbers=288, seed=n) 

    precision, recall, auc_precision_recall = node_anomaly_gator.get_PR_AUC(node_anomaly_gator.get_true_anomaly_label_core_struc(), node_anomaly_gator.get_cora_prediction(), title_PR_type="Node Anomaly on Cora dataset")

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
labels = node_anomaly_gator.get_true_anomaly_label_core_struc()
positive_ratio = labels.count(1)/len(labels)
plt.hlines(positive_ratio, xmin=0, xmax=1, color='red', label='Baseline')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Node anomaly\nMean Precision-Recall Curve with Std (alliGATOR)")
plt.legend()
# plt.savefig("Cora_node_anomaly_2025-05-20_11-34-25_779_100_45")
plt.show()

# node_anomaly_gator.plot_active_edges_and_nodes()

scores = node_anomaly_gator.get_cora_prediction()
labels = node_anomaly_gator.get_true_anomaly_label_core_struc()
scores = np.array(scores)
labels = np.array(labels)
plt.hist([scores[labels==0], scores[labels==1]], bins=50, color=['blue', 'red'], label=['Normal', 'Anomalous'])
plt.title("Node anomaly score distribution")
plt.legend()
plt.show()

anomalous_graphs = [c for c, i in enumerate(node_anomaly_gator.get_true_anomaly_label_core_struc()) if i == 1]
for g in [1, 2, 8, 6, 209]:
    node_anomaly_gator.plot_graph_Cora(g, plot_only_existing_edges=True)
    node_anomaly_gator.plot_graph_Cora(g, plot_only_existing_edges=False)