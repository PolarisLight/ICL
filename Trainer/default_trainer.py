"""
This file contains the default trainer class
Default_Trainer contains the following functions:
    __init__: initialize the trainer
    prepare_folder: prepare the folder for saving the model
    save_checkpoint: save the model checkpoint
    load_checkpoint: load the model checkpoint
    train_one_epoch: train the model for one epoch
    _eval: evaluate the model
    _on_epoch_end: print the results of the epoch
    train: train the model
    predict: predict the model
    adjust_learning_rate: adjust the learning rate
Other trainers can be defined by inheriting from Default_Trainer
"""
import torch, time, math
from tqdm import tqdm
from loguru import logger
import os
from config.Arguments import Arguments
from utils.schedular import get_scheduler
import wandb

def log_training_config(args, base_config):
    for k, v in vars(args).items():
        if k == 'base_args':
            continue
        if isinstance(v, Arguments):
            for kk, vv in vars(v).items():
                base_value = getattr(getattr(base_config, k), kk, None)
                if vv != base_value:
                    logger.bind(params=True).info(f"**Modified** {k}.{kk}: {vv} (Base: {base_value})")
                else:
                    logger.bind(params=True).info(f"{k}.{kk}: {vv}")
        else:
            base_value = getattr(base_config, k, None)
            if v != base_value:
                logger.bind(params=True).info(f"**Modified** {k}: {v} (Base: {base_value})")
            else:
                logger.bind(params=True).info(f"{k}: {v}")

def init_weights(model, method='kaiming', exclude=[]):
    for name, param in model.named_parameters():
        if not any(name.startswith(e) for e in exclude):
            if 'weight' in name:
                if method == 'kaiming':
                    torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif method == 'xavier':
                    torch.nn.init.xavier_normal_(param)
                else:
                    raise ValueError(f"Unknown weight initialization method: {method}")
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)

def print_trainable_parameters(model):
    print("Model structure:")
    total_params = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    # print total number of parameters with human readable format, i.e., 1k, 1M
    print(f"Total number of trainable parameters: {total_params:,}")

class Default_Trainer():
    def __init__(self,args,model,dataset,device):
        self.args = args   
        self.start_epoch = 0
        if hasattr(args,'base_args'):
            log_training_config(args,base_config=args.base_args)
        self.model = model
        # print_trainable_parameters(self.model)
        self.dataset = dataset
        self.device = device
        self.model.to(self.device)

        if self.args.dataset.loader is not None:
            self.loader = {'train': self.dataset.train_loader, 'val': self.dataset.val_loader}
        else:
            self.loader = {'train': torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.args.train.batch_size, shuffle=True, num_workers=self.args.train.num_workers),
                           'val': torch.utils.data.DataLoader(self.dataset['val'], batch_size=self.args.train.batch_size, shuffle=False, num_workers=self.args.train.num_workers),}
        
        
        try:
            self.optimizer = getattr(torch.optim, self.args.train.optimizer)(self.model.parameters(), lr=self.args.train.lr, weight_decay=self.args.train.weight_decay)
        except AttributeError:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        try :
            self.scheduler = get_scheduler(args=self.args)
        except NotImplementedError:
            self.scheduler = None
            logger.info("No learning rate scheduler")

        self.prepare_folder()
        if args.train.use_wandb:
            wandb.init(project=args.task, name=args.train.save_name)
            wandb.config.update(args)
            wandb.watch(self.model)

        
    def prepare_folder(self):
        """
        Prepare the folder for saving the model
        """
        if self.args.train.save_log:
            if not os.path.exists(self.args.train.save_dir):
                os.makedirs(self.args.train.save_dir)
            self.save_path = os.path.join(self.args.train.save_dir, self.args.train.save_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)


    def save_checkpoint(self,epoch,filename='checkpoint.pt'):
        """
        Save the model checkpoint
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_epoch': self.best_epoch,
        }
        torch.save(checkpoint, os.path.join(self.save_path, filename))

    def load_checkpoint(self):
        """
        Load the model checkpoint
        """
        if self.args.train.resume is not None:
            checkpoint = torch.load(self.args.train.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['best_epoch']
            pre_val_results = self._eval(mode='val',epoch=checkpoint['epoch'])
            pre_val_results = {**pre_val_results['loss'], **pre_val_results['metric']}
            self.best_val_metrics = pre_val_results[self.args.train.core_metric[0]]
            logger.info(f"Checkpoint loaded at epoch {self.start_epoch}, val metrics {self.args.train.core_metric[0]}:{self.best_val_metrics:.4f}")
        else:
            logger.info("No checkpoint loaded")

    def train_one_epoch(self,epoch):
        """
        Train the model for one epoch
        """
        self.model.train()
        with tqdm(self.loader['train'], desc=f"Epoch {epoch}", unit='batch') as pbar:
            for i,data in enumerate(pbar):
                self.optimizer.zero_grad()
                outputs = self.model.train_step(data)
                outputs['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.train.clip_grad)
                self.optimizer.step()

                pbar.set_postfix({'loss': outputs['loss'].item()})
                if self.args.train.use_wandb and i % self.args.train.log_interval == 0:
                    wandb.log({'batch':epoch*len(self.loader['train'])+i, 'train loss': outputs['loss'].item()})
                 
        return outputs['loss'].item()
    
    def _eval(self,mode='val'):
        """
        Evaluate the model
        """
        self.model.eval()
        if mode == 'val':
            loader = self.loader['val']
        elif mode == 'test':
            loader = self.loader['test']
        else:
            raise ValueError(f"Unknown mode: {mode}")
        pred_list = []
        target_list = []
        with torch.no_grad():
            with tqdm(loader, desc=f"Evaluating {mode}", unit='batch') as pbar:
                for i,data in enumerate(pbar):
                    outputs = self.model.eval_step(data)
                    pred_list.append(outputs['pred'])
                    target_list.append(data['label'])

        results = self.model.on_eval_end(pred_list, target_list)
        return results
    
    def _on_epoch_end(self, epoch, val_metrics,goal):
        """
        Print the results of the epoch
        """
        if self.args.train.save_log:
            if not hasattr(self.args.train,'not_save_ckpt'):
                    
                # save the model
                if hasattr(self.args.train,'save_freq'):
                    if epoch % self.args.train.save_freq == 0:
                        self.save_checkpoint(epoch,f'checkpoint_{epoch}.pt')
                        print(f"Model saved at epoch {epoch}")
                elif hasattr(self.args.train,'save_epoch'):
                    if epoch in self.args.train.save_epoch:
                        self.save_checkpoint(epoch,f'checkpoint_{epoch}.pt')
                        print(f"Model saved at epoch {epoch}")
                # save the best model
                if goal == 'maximize':
                    if val_metrics > self.best_val_metrics:
                        self.best_val_metrics = val_metrics
                        self.best_epoch = epoch
                        self.save_checkpoint(epoch,f"best_checkpoint.pt")
                        logger.info(f"Best model saved at epoch {epoch}, metric {self.args.train.core_metric[0]}: {val_metrics}")
                elif goal == 'minimize':
                    if val_metrics < self.best_val_metrics:
                        self.best_val_metrics = val_metrics
                        self.best_epoch = epoch
                        self.save_checkpoint(epoch,f"best_checkpoint.pt")
                        logger.info(f"Best model saved at epoch {epoch}, metric {self.args.train.core_metric[0]}: {val_metrics}")
        else:   
                if goal == 'maximize':
                    if val_metrics > self.best_val_metrics:
                        self.best_val_metrics = val_metrics
                        self.best_epoch = epoch
                elif goal == 'minimize':
                    if val_metrics < self.best_val_metrics:
                        self.best_val_metrics = val_metrics
                        self.best_epoch = epoch

        # early stopping
        if epoch - self.best_epoch > self.args.train.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            return True
        return False


    def train(self):
        """
        Train the model
        """
        self.best_val_metrics = -math.inf if self.args.train.core_metric[1] == 'maximize' else math.inf
        self.best_epoch = 0
        self.load_checkpoint()
        for epoch in range(self.start_epoch,self.args.train.epochs):
            self.adjust_learning_rate(epoch, self.args.train.lr)
            start_time = time.time()
            _ = self.train_one_epoch(epoch=epoch)
            val_result = self._eval(mode='val')
            end_time = time.time()
            val_dict = {'epoch':epoch,**val_result['loss'],**val_result['metric']}
            if self.args.train.use_wandb:
                wandb.log(val_dict)
            for key,value in val_dict.items():
                if isinstance(value,float):
                    val_dict[key] = f"{value:.4f}"
            logger.info(f"Epoch {epoch}[{end_time-start_time:.2f}s]: {val_dict}")
            if self._on_epoch_end(epoch, float(val_dict[self.args.train.core_metric[0]]),self.args.train.core_metric[1]):
                break
        if hasattr(self.loader,'test'):
            self._eval(mode='test')
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best Metric: {self.best_val_metrics}")
        return_dict = {'best_epoch':self.best_epoch, 'best_metric':self.best_val_metrics}
        return return_dict
    
    def predict(self):
        """
        Predict the model
        """
        self.load_checkpoint()
        test_result = self._eval(mode='test')
        return test_result['pred']
    
    def adjust_learning_rate(self, epoch, lr):
        """
        Adjust the learning rate
        """
        if self.scheduler is None:
            return lr
        new_lr = self.scheduler(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        if new_lr != lr and self.args.scheduler.name == 'step':
            logger.info(f"[Epoch {epoch}] Update learning rate from {lr:.2e} to {new_lr:.2e}")
        
        



            
        
            
        
