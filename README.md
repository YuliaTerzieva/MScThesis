# MScThesis

This is the official implementation of [alliGATOR : Graph Anomaly Detection through diffusion based Topology Reconstruction](https://studenttheses.uu.nl/handle/20.500.12932/49344)

Thesis defence presentation : [alliGATOR - Thesis defence.pdf](<alliGATOR - Thesis defence.pdf>)

## Abstract

Financial crime detection concerning monetary transactions remains a critical challenge in the
fight against illicit activities such as fraud and money laundering. In this thesis, we present the
alliGATOR model, an unsupervised graph anomaly detection model designed to identify both node
and edge anomalies. Our approach is grounded in the notion that node connections are dependent
on their attributes. We hypothesised that anomalies can be detected by comparing a graph’s original
topology to one reconstructed from node attributes alone.

The alliGATOR model requires a novel node-guided topology reconstructor model – a discrete
generative diffusion model based on a graph neural network link predictor. The model iteratively
reconstructs the graph structure by adding edges conditioned on the node attributes and degrees,
generating a topology that obeys the local properties of the nodes. Our node-guided topology
reconstructor operates under an inductive setting and can generalise across graphs.

We evaluate the alliGATOR model on a custom synthetic dataset designed to simulate real-world
behaviour, and compare its performance to the Isolation Forest baseline. Experimental results show
that alliGATOR is robust and successfully identifies both node and edge anomalies.

This work establishes a foundation for applying generative diffusion models to graph anomaly de-
tection, and it can support researchers, financial institutions, banks, and governments in developing
personalised autonomous anomaly detection systems.

## In this repository you can find

### Dataset generation (Chapter 4.1)
To generate a dataset with custom requirements consult the [DatasetGraphGenerator.py](DatasetGraphGenerator.py) file.

The code first generates a whole graphs (saved in [GeneratedDataset_interm_graph](GeneratedDataset_interm_graph)), and then precomputes the subgraphs around each instance on interest (saved in [GeneratedDataset](GeneratedDataset))

### Training the node-guided topology reconstructor model using the generated data (Chapter 3.2.1):
**Note**: Don't forget to cd to the EDGE folder before you move on. 

The node-guided topology reonstructor is built on the [EDGE model by Chen et al](https://github.com/tufts-ml/graph-generation-EDGE). 

To train a new reconstructor run [EDGE/train.py](EDGE/train.py) with parameters of choice. 
An example is presented below for the edge experiment :

```
python3 train.py
    --epochs 1000  
    --diffusion_dim 16 
    --diffusion_steps 5 
    --device cpu 
    --dataset Synthetic_K7_edge 
    --dim_node_attr 3
    --batch_size 32 
    --lr 1e-3 
    --p_uncon 0.2 
    --optimizer adam 
    --final_prob_edge 1 0 
    --sample_time_method uniform 
    --check_every 3 
    --eval_every 3 
    --noise_schedule linear 
    --dp_rate 0.1 
    --degree 
    --num_heads 3 1 
```

The trained model will be saved in [EDGE/wandb](EDGE/wandb)

### Using the alliGATOR model (Chapter 3.2.2 and 3.2.3)
After the model is trained, use the alliGATOR model (implemented in [EDGE/alliGATOR.py](EDGE/alliGATOR.py)) by running 
[EDGE/node_anomaly_detection.py](EDGE/node_anomaly_detection.py) for node anomaly detection, and [EDGE/edge_anomaly_detection.py](EDGE/edge_anomaly_detection.py) for edge anomaly detection. 

Hyperparameter tuning is possible after the individual topology reconstructors are trained. 

For node anomaly hyperparameter tuning consult [EDGE/node_anomaly_hyperparam_tuning.py](EDGE/node_anomaly_hyperparam_tuning.py). 

for edge anomaly hyperparameter tuning consult [EDGE/edge_anomaly_hyperparam_tuning.py](EDGE/edge_anomaly_hyperparam_tuning.py).

### Isolation forest (Chapter 4.3.1)

The experiment for Isolation forest are presented in [Isolation forest](<Isolation forest>) -- [ for node anomaly](<Isolation forest/isolation_forest_node_anomaly.py>) and [for edge anomaly](<Isolation forest/isolation_forest_edge_cls.py>)

