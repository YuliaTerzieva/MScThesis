import numpy as np
import argparse
from diffusion.utils import add_parent_path, set_seeds

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Exp
from experiment import GraphExperiment, add_exp_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion.optim.multistep import get_optim, get_optim_id, add_optim_args

# Added by Yulia because I work on M3
import torch.multiprocessing




if  __name__ == '__main__': # Yulia : I added this because otherwise I get a multiprocessing spawn error
    torch.multiprocessing.set_start_method('spawn', force=True)

    ###########
    ## Setup ##
    ###########
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_exp_args(parser)
    add_model_args(parser)
    add_optim_args(parser)
    args = parser.parse_args()
    set_seeds(args.seed)

    ##################
    ## Specify data ##
    ##################

    train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

    args.num_edge_classes = num_edge_classes
    args.num_node_classes = num_node_classes

    if args.final_prob_node is None:
        args.final_prob_node = [1-1e-12, 1e-12]
        args.num_node_classes = 2
        args.has_node_feature = False

    if 0 in args.final_prob_edge:
        args.final_prob_edge[np.argmax(args.final_prob_edge)] = args.final_prob_edge[np.argmax(args.final_prob_edge)]-1e-12
        args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

    args.max_degree = max_degree
    args.num_node_feat = num_node_feat
    args.augmented_feature_dict = augmented_feature_dict



    data_id = get_data_id(args)
    ###################
    ## Specify model ##
    ###################

    model = get_model(args, initial_graph_sampler=initial_graph_sampler)
    model_id = get_model_id(args)
    #######################
    ## Specify optimizer ##
    #######################

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    optim_id = get_optim_id(args)

    ##############
    ## Training ##
    ##############
    exp = GraphExperiment(args=args,
                    data_id=data_id,
                    model_id=model_id,
                    optim_id=optim_id,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    test_loader=test_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler_iter=scheduler_iter,
                    scheduler_epoch=scheduler_epoch,
                    monitoring_statistics=monitoring_statistics,
                    eval_evaluator=eval_evaluator, 
                    test_evaluator=test_evaluator,
                    n_patient=50)

    exp.run()


    """
    To run : 
    
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --epochs 1 --num_generation 64 --diffusion_dim 64 --diffusion_steps 32 --device mps --dataset Ego --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 
    
    
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --epochs 1 --num_generation 64 --diffusion_dim 64 --diffusion_steps 32 --device mps --dataset Ego_Nets_conf3 --batch_size 4 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 1 --eval_every 1 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt_prescribred_st --arch TGNN_degree_guided --parametrization xt_prescribed_st --empty_graph_sampler empirical --degree --num_heads 8 8 8 8 1 

    """

    """
    I am staeting to hate mac books ... NotImplementedError: The operator 'aten::scatter_reduce.two_out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    
    """