# MScThesis

## Generating new data
To generate a dataset with custom requirements, add a new configuration to "configurations.json" and call DataGeneration.py as such:

```
python3 DataGeneration.py --setting "_the_name_of_your_configuration_"
```

Currently, the code can generate 3 types of datasets :

#### Dataset type 1
Based on a circular pattern of blue nodes. Each blue node is also connected to an orange node, which in turn has a grey node connected to t. Each grey node is connected to two orange nodes, that are connected to two neighbouring blue nodes. Below is an example of such a graph. After the pattern is repeated _"pattern_number"_ of times, there are _"new_connections"_ number of new edges added between the patterns randomly.

#### Dataset type 2
Based on a pattern of 6 nodes: 4 blue nodes pointing to a single orange, which points to another orange node. After the pattern is repeated _"pattern_number"_ of times, there are _"new_connections"_ number of new edges added between the patterns randomly. The nodes that become connected and noted/flagged as anomalies. The last step of the data generation is injecting _"number_random"_ number gray nodes that are connected to the rest of the nodes randomly. The degree of those nodes are drawn from a Gaussian distribution with _"mu"_ and _"std"_.

Calling the python file as shown above would create 4 files in the GeneratedDataset folder: two CSV files with nodes and edges, respectively, and two pickles with a list of networkx ego graphs in each. One is with 90% of the ego networks and can be used for training; the other one is with the other 10% of the ego networks and can be used for testing. Although the Big network is directed, the ego_networks that are saved in the pickle are **undirected**

For all the datasets typed, hop is the number of hops used when creating the ego networks. 

#### Dataset type 3
_In the making ..._

## Training the EDGE model using the generated data:
**Note**: Don't forget to cd to the EDGE folder before you move on. 
In EDGE/train.py I have commented a few examples of how to train a model with my datasets

I've made the following changes to the original EDGE code:
- The code can now work with my datasets; for this, I have changed EDGE/datasets/data.py and EDGE/datasets/data_utils.py. 
- I have added a new _empty_graph_sampler_ called **EmptyGraphGeneratorWithNodeAttributes**, which samples graphs from the testing dataset pickle (created in DataGeneration.py)
- _to be continued_
