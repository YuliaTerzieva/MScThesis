# MScThesis

## Generating new data
To generate a dataset with custom requirenments, add a new configuration to "configurations.json" and call DataGeneration.py as such:

'''
python3 DataGeneration.py --setting "_the_name_of_your_configuration_"
'''

Currently, the code can generate datasets that are all based on a pattern of 6 nodes : 4 blue nodes pointing to a single orange, which points to another orange node. After the pattern is repeated _"pattern_number"_ of times, there are _"new_connections"_ number of new edges added between the patterns randomly. The nodes that become connected and noted/flagged as anomalies. The last step of the datageneration is injecting _"number_random"_ number gray nodes, that are connected to the rest of the nodes randomly. The degree of those nodes are drawn from a gaussian distribution with _"mu"_ and _"std"_.

Calling the python file as shown above would create 3 files in the GeneratedDataset foulder : two csv files with nodes and edges respectively, and a pickle with a list of networkx ego graphs. Althought the Big network is directed, the ego_networks that are saved in the pickle are **undirected**

## Training the EDGE model using the generated data:

Following the instruction of EDGE ReadMe file, to train the model please do the following:

'''
cd EDGE
python3 train.py --epochs 50 --num_generation 32 --diffusion_dim 32 --diffusion_steps 32 --device cuda:1 --dataset _"the name of your dataset"_ --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 
'''

For example, I have created a dense 2 hop ego network using configuration _very_graph_non_directional_ and the dataset name is _Big_Ego_Nets_non_dir_. Thus calling the EDGE model with this dataset would be :

'''
python3 train.py --epochs 50 --num_generation 32 --diffusion_dim 32 --diffusion_steps 32 --device cuda:1 --dataset Big_Ego_Nets_non_dir --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 
'''

**Note**: don't forget to cd to the EDGE foulder. 



