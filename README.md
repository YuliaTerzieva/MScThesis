# MScThesis

## Generating dataset
To generate a dataset with custom requirements consult the [DatasetGraphGenerator.py](DatasetGraphGenerator.py) file.
After the code is run, it generates the whole graphs in [GeneratedDataset_interm_graph](GeneratedDataset_interm_graph) and then precomputes the subgraphs around each instance on interest and saves them in [GeneratedDataset](GeneratedDataset)

## Training the node-guided topology reconstructor (adapted EDGE) model using the generated data:
**Note**: Don't forget to cd to the EDGE folder before you move on. 

The node-guided topology reonstructor is built on the EDGE model by Chen et al. 

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

## Using the alliGATOR model
After the model is trained, use the alliGATOR model (implemented in [EDGE/alliGATOR.py](EDGE/alliGATOR.py)) by running 
[EDGE/node_anomaly_detection.py](EDGE/node_anomaly_detection.py) for node anomaly detection, and [EDGE/edge_anomaly_detection.py](EDGE/edge_anomaly_detection.py) for edge anomaly detection. 

Hyperparameter tuning is possible after the individual topology reconstructors are trained. 
For node anomaly hyperparameter tuning consult [EDGE/node_anomaly_hyperparam_tuning.py](EDGE/node_anomaly_hyperparam_tuning.py). 
for edge anomaly hyperparameter tuning consult [EDGE/edge_anomaly_hyperparam_tuning.py](EDGE/edge_anomaly_hyperparam_tuning.py).

## The Isolation forest

The experiment for Isolation forest are presented in [Isolation forest](<Isolation forest>) -- [ for node anomaly](<Isolation forest/isolation_forest_node_anomaly.py>) and [for edge anomaly](<Isolation forest/isolation_forest_edge_cls.py>)

