import torch
from datetime import datetime
from loguru import logger
from config.Arguments import Arguments
import os
## local paths
root_dir = './' 
label_dir = 'Datasets/ltvr/cifar100'
data_dir = 'Datasets/ltvr/cifar100' 
utils_dir = './'

## server paths
save_dir = './Log/'

def config(task:str, model:str, dataset:str, opt:Arguments=None)->Arguments:
    """
    Configure the arguments for the training
    task: task name
    model: model name
    dataset: dataset name
    opt: additional arguments
    """
    args = Arguments()
    args.task = task

    args.model = Arguments()
    args.train = Arguments()
    args.dataset = Arguments()
    args.scheduler = Arguments()
    args.rt_scheduler = Arguments()
    
    # training parameters
    args.train.batch_size = 128
    args.train.epochs = 200
    args.train.lr = 0.05
    args.train.lr_rt = 0.15
    args.train.rt_start = 180 
    args.train.optimizer = 'SGD'
    args.train.rt_optimizer = 'SGD'
    args.train.weight_decay = 5e-4
    args.train.num_workers = 0 
    args.train.save_dir = os.path.join(save_dir, task)
    args.train.save_name = f'{dataset}_{model}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    args.train.resume = None # 'Log/shike-moe/base/checkpoint_179.pt'
    args.train.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.train.use_wandb = 1
    args.train.clip_grad = 5
    args.train.log_interval = 100
    args.train.save_epoch = [119,179]
    args.train.early_stop = 200
    args.train.core_metric = ['acc1','maximize']

    # learning rate scheduler
    
    args.scheduler.name = 'cosineannel'
    args.scheduler.warmup_epoch = 5
    args.scheduler.total_epoch = args.train.epochs-20
    args.scheduler.eta_min = 0.0
    args.scheduler.base_lr=args.train.lr
    args.scheduler.warmup_lr=0.15

    args.rt_scheduler.name = 'cosineannel'
    args.rt_scheduler.warmup_epoch = 5
    args.rt_scheduler.total_epoch = 20
    args.rt_scheduler.eta_min = 0.0
    args.rt_scheduler.base_lr=args.train.lr
    args.rt_scheduler.warmup_lr=0.1

    ## parameters for dataset

    args.dataset.name = dataset
    args.dataset.label_dir = label_dir
    args.dataset.data_dir = data_dir
    args.dataset.loader = None
    args.dataset.imgsz = 32
    args.dataset.imb_factor = 0.01

    ## parameters for model
    args.model.name = model
    args.model.num_classes = 100 if 'cifar100' in dataset.lower() else 10
    args.model.dropout = 0.1

    args.core_params = Arguments()
    args.core_params.ce_weight = 1
    args.core_params.distill_weight = 1
    args.core_params.parallel_weight = 1
    args.core_params.diversity_epoch = 30
    args.core_params.diversity_weight = 1
    args.core_params.hnm_weight = 1
    args.core_params.hnm_start_epoch = 120
    args.core_params.hnm_k = 7
    args.core_params.hnm_momentum = 0.9
    args.core_params.hnm_temperature = 0.07

    return args
    