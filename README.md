# MScThesis

## Generating new data
To generate a dataset with custom requirenments, add a new configuration to "configurations.json" and call DataGeneration.py as such:

```
python3 DataGeneration.py --setting "_the_name_of_your_configuration_"
```

Currently, the code can generate datasets that are all based on a pattern of 6 nodes : 4 blue nodes pointing to a single orange, which points to another orange node. After the pattern is repeated _"pattern_number"_ of times, there are _"new_connections"_ number of new edges added between the patterns randomly. The nodes that become connected and noted/flagged as anomalies. The last step of the datageneration is injecting _"number_random"_ number gray nodes, that are connected to the rest of the nodes randomly. The degree of those nodes are drawn from a gaussian distribution with _"mu"_ and _"std"_.

Calling the python file as shown above would create 3 files in the GeneratedDataset foulder : two csv files with nodes and edges respectively, and a pickle with a list of networkx ego graphs. Althought the Big network is directed, the ego_networks that are saved in the pickle are **undirected**

## Training the EDGE model using the generated data:

Following the instruction of EDGE ReadMe file, to train the model please do the following:

```
cd EDGE
python3 train.py --epochs 50 --num_generation 32 --diffusion_dim 32 --diffusion_steps 32 --device cuda:1 --dataset _"the name of your dataset"_ --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 
```

For example, I have created a dense 2 hop ego network using configuration _very_graph_non_directional_ and the dataset name is _Big_Ego_Nets_non_dir_. Thus calling the EDGE model with this dataset would be :

```
python3 train.py --epochs 50 --num_generation 32 --diffusion_dim 32 --diffusion_steps 32 --device cuda:1 --dataset Big_Ego_Nets_non_dir --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 
```

**Note**: don't forget to cd to the EDGE foulder. 

### To do:
- discuss with Vahid and Ramon how to add the node attributes (in our case color) to the model, as the model creates node embeddings only using the relation between nodes. Maybe I can add it there?
    - check the last paragraph of section 2.5 in the paper; they use MPNN to compute node representations Zt, I can alter that to take the node color into account! 
    - discuss solution with Salvatore
    - maybe do it in data_utila somewhere in lines 58 to 61 :) 
- save the trained model and try to generate edges in an empty graph
- train the model for smaller graphs (possbly won't work as the authors don't answer my email)
- consider making it directional (possibly impossible :D)
- set up a pipeline for checking for anomaly
- work on similarity measure/scoring mechanism for anomaly score
- minimal vaible product alliGATOR

- generate the two other datasets
    - simpler pattern only
    - more complex relation based one

- set seed when passing arguments 
- check number of nodde classes and features in datasets/data.py
- check final_prob_node + edge ??? 
    - My understanding : in the exmaple call the argument --final_prob_edge 1 0 is present. from my understanding that means there are two types of edges : "there is no edge", "there is edge". 
    the probability of the two classes should some to one and the ones given are at time step T "empty graph", which here is probability of 1 for there to be no edges and 0 of there being edges.

- check in model.py line 50 : num_edge_classes=args.num_node_classes ??? why, is this a mistake?


---- 
ToDo from 24th :
- Why the number of forward passes are less than the diffusion steps? what is the early quitting queteria? -> Answer : it is dependent on the active edge indices in p_sample in diffusion/diffusion_binomial_active.py
- I need to change the _sample_graph_size_and_features, instead of having a random empty graph, I need to pass my nodes and node attriutes. However, i need to figure out how to match them...
- In diffusion_base.py sample -> t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long) # Yulia: FYI this is not used at all, why do we have it

full edge index are all the possible undirected edges without self-loops that can exist in a graph with number_of_nodes nodes!!!

---- end of day recap : I figured what the problem was with the diffusion steps - the linear noise schedule scaling, i changed it to cosine instead of fixing their linear function, because cosine is better either way. The other big thing is that i used pbd and i finally know where the model is being called!

Now on the todo list is to train the model with node features and try to test call it to generate edges on empthy graphs of your choice!
----
ToDo from the 25th : 
- how do they decide which are the active nodes? -> check paper
    -> testing the code with pdb it seems that you have can the same node active twice in a row

- make the the color attribute be node_attr and possibly one-hot encoding?
- Question for Ioana : how big are the working graphs (egonetworks) in the bank actually? are we talking tens, hungreds or thousands of nodes? how many edges?

---
ToDO fom 26 of Feb:
- check if p in formula 1 form the paper -> check if p is the same or they are distinct