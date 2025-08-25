"""
This is the main file for training CIFAR dataset.
This file is used to test this framework.
"""
from argparse import Namespace, ArgumentParser
from utils.utils import set_seed
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# need to be redefined for every new task
from config.config_cifar_base import config as base_config
from config.config_cifar_moe import config as config
import models.Resnet as models
import datasets.CIFAR_LT as datasets
# =============================================================================

def get_model_and_dataset(args:Namespace)->tuple:
    """
    Get the model and dataset.
    This function need to be redefined for every new task.
    """
    
    dataset = {'train': getattr(datasets, args.dataset.name)(args, train=True),
               'val':getattr(datasets, args.dataset.name.replace("IMBALANCE",""))(args, train=False)}

    args.label_dis = dataset['train'].get_cls_num_list()
    model = getattr(models, args.model.name)(args)
    return model, dataset

def main(args:Namespace)->None:
    """
    Main function
    args: arguments
    """
    model, dataset = get_model_and_dataset(args)
    
    from Trainer.moe_trainer import MoE_Trainer
    trainer = MoE_Trainer(args, model, dataset, args.train.device)
    trainer.train()
    

# =============================================================================
# Normal Mode
# =============================================================================

if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='ICL', help='task name')
    opt.add_argument('--model', type=str, default='ResNet_MoE', help='model name')
    opt.add_argument('--dataset', type=str, default='IMBALANCECIFAR100', help='dataset name')
    opt.add_argument('--seed', type=int, default=123, help='random seed')
    opt.add_argument('--save_log', type=bool, default=True, help='save log')
    opt = opt.parse_args()
    set_seed(opt.seed)
    args = config(opt.task, opt.model, opt.dataset, opt.save_log)
    args.base_args = base_config(opt.task, opt.model, opt.dataset, opt.save_log)
    main(args)
