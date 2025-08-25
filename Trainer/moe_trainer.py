import torch
from loguru import logger
from Trainer.default_trainer import Default_Trainer,log_training_config,print_trainable_parameters
from utils.schedular import get_scheduler,CosineAnnealingLRWarmup
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast as autocast, GradScaler
from config.Arguments import Arguments
import nni

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MoE_Trainer(Default_Trainer):
    def __init__(self,args,model,dataset,device):
        self.args = args   
        self.start_epoch = 0
        if hasattr(args,'base_args'):
            log_training_config(args,base_config=args.base_args)

        self.model = model
        print_trainable_parameters(self.model)
        self.dataset = dataset
        self.device = device
        self.model.to(self.device)

        if self.args.dataset.loader is not None:
            self.loader = {'train': self.dataset.train_loader, 'val': self.dataset.val_loader}
        else:
            weights_per_class = 1.0 /torch.tensor(args.label_dis) 
            weights_per_class /= weights_per_class.sum()
            train_labels = torch.argmax(torch.tensor(self.dataset['train'].targets),dim=1)  
            sample_weights = weights_per_class[train_labels]
            
            self.sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(self.dataset['train']), replacement=True)

            self.loader = {'train': torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.args.train.batch_size, 
                                                                 num_workers=self.args.train.num_workers,
                                                                # sampler=self.sampler,
                                                                shuffle=True,
                                                                ),
                           'val': torch.utils.data.DataLoader(self.dataset['val'], batch_size=self.args.train.batch_size, 
                                                              shuffle=False, num_workers=self.args.train.num_workers),}
        

        self.scaler = GradScaler()
        self.optimizer = getattr(torch.optim, self.args.train.optimizer)(self.model.parameters(), 
                                                                             lr=self.args.train.lr, 
                                                                             weight_decay=self.args.train.weight_decay,
                                                                             momentum=0.9)
  
        
        
        self.rt_optimizer = getattr(torch.optim, self.args.train.rt_optimizer)(self.model.parameters(), 
                                                                                   lr=self.args.train.lr_rt if hasattr(self.args.train,'lr_rt') else self.args.train.lr,
                                                                                   weight_decay=self.args.train.weight_decay,
                                                                                   momentum=0.9)
        
        self.scheduler = CosineAnnealingLRWarmup(
            optimizer=self.optimizer,
            T_max=self.args.train.rt_start,
            eta_min=0.0,
            warmup_epochs=5,
            base_lr=self.args.train.lr,
            warmup_lr=0.15
        )
        
        self.rt_scheduler = CosineAnnealingLRWarmup(
            optimizer=self.rt_optimizer,
            T_max=self.args.train.epochs - self.args.train.rt_start,
            eta_min=0.0,
            warmup_epochs=5, #TODO：ori 5
            base_lr=self.args.train.lr_rt if hasattr(self.args.train,'lr_rt') else self.args.train.lr,
            warmup_lr=0.1
        )

        
        self.prepare_folder()
        if args.train.use_wandb:
            wandb.init(project=args.task, name=args.train.save_name)
            wandb.config.update(args)
            wandb.watch(self.model)
        

    def train_one_epoch(self, epoch):
        self.model.train()
        if epoch >= self.args.train.rt_start:
            optimizer = self.rt_optimizer
        else:
            optimizer = self.optimizer
        with tqdm(self.loader['train'], desc=f"Epoch {epoch}", unit='batch') as pbar:
            for i, data in enumerate(pbar):

                optimizer.zero_grad()

                with autocast():
                    outputs = self.model.train_step(data, epoch >= self.args.train.rt_start,epoch)
                loss = outputs['loss']
                self.scaler.scale(loss).backward()

                self.scaler.step(optimizer)
                self.scaler.update()

                if i % self.args.train.log_interval == 0:
                    pbar.set_postfix({'loss': loss.item()})
                    if self.args.train.use_wandb:
                        wandb.log({'batch': epoch * len(self.loader['train']) + i, 'training loss': loss.item()})
        if epoch >= self.args.train.rt_start:
            self.rt_scheduler.step()
        else:
            self.scheduler.step()
        return loss.item()
    
    def _eval(self,mode='val',epoch=0):
        """
        Evaluate the model
        """
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        exp_acc= []

        self.model.eval()
        if mode == 'val':
            loader = self.loader['val']
        elif mode == 'test':
            loader = self.loader['test']
        else:
            raise ValueError(f"Unknown mode: {mode}")
        with torch.no_grad():
            with tqdm(loader, desc=f"Evaluating {mode}", unit='batch') as pbar:
                for _,data in enumerate(pbar):
                    outputs = self.model.eval_step(data,rt=epoch>=self.args.train.rt_start)
                    losses.update(outputs['loss'].item(), data['image'].size(0))
                    top1.update(outputs['acc1'], data['image'].size(0))
                    top5.update(outputs['acc5'], data['image'].size(0))
                    exp_acc.append(outputs['exp_acc'])

                exp_acc = torch.stack(exp_acc)
                exp_acc = torch.mean(exp_acc,dim=0)
                exp_acc = exp_acc.cpu().numpy()
                exp_acc = exp_acc.tolist()
        if hasattr(self,'save_path'):
            import csv
            with open(f"{self.save_path}/{mode}_results.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(exp_acc)
            if not epoch>=self.args.train.rt_start:
                with open(f"{self.save_path}/{mode}_grad.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([grad.item() for grad in self.gradients_experts])
        self.model.on_eval_end()
        results = {'loss': {'ce':losses.avg}, 'metric': {'acc1': top1.avg.item(), 'acc5': top5.avg.item()}}
        return results
    

    def train(self):
        
        """
        Train the model
        """
        import math
        import time
        self.best_val_metrics = -math.inf if self.args.train.core_metric[1] == 'maximize' else math.inf
        self.best_epoch = 0
        self.load_checkpoint()
        for epoch in range(self.start_epoch,self.args.train.epochs):
            if epoch >= self.args.train.rt_start:
                for name, param in self.model.named_parameters():
                    if name[:14] != "rt_classifiers":
                        param.requires_grad = False
            
            self.adjust_learning_rate(epoch, self.args.train.lr)
            start_time = time.time()
            train_loss = self.train_one_epoch(epoch=epoch)
            
            val_result = self._eval(mode='val',epoch=epoch)
            end_time = time.time()
            val_dict = {'epoch':epoch,**val_result['loss'],**val_result['metric']}
            if self._on_epoch_end(epoch, float(val_dict[self.args.train.core_metric[0]]),self.args.train.core_metric[1]):
                break
            val_dict['best metric'] = self.best_val_metrics
            if self.args.train.use_wandb:
                wandb.log(val_dict)
            if hasattr(self.args.train,'use_nni') and self.args.train.use_nni:
                nni.report_intermediate_result({'default':float(val_dict[self.args.train.core_metric[0]]),
                                                'cur_best':val_dict['best metric'],
                                                })
            for key,value in val_dict.items():
                if isinstance(value,float):
                    val_dict[key] = f"{value:.4f}"
            logger.info(f"Epoch {epoch}[{end_time-start_time:.2f}s]: {val_dict}")
            
        if hasattr(self.loader,'test'):
            self._eval(mode='test')
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best Metric: {self.best_val_metrics}")
        return_dict = {'best_epoch':self.best_epoch, 'best_metric':self.best_val_metrics}
        if hasattr(self.args.train,'use_nni') and self.args.train.use_nni:
            nni.report_final_result({'default':self.best_val_metrics})
        return return_dict
    
    def adjust_learning_rate(self, epoch, lr):
        pass

    def save_checkpoint(self,epoch,filename='checkpoint.pt'):
        """
        Save the model checkpoint
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rt_optimizer': self.rt_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'rt_scheduler': self.rt_scheduler.state_dict(),
            'epoch': epoch,
            'best_epoch': self.best_epoch,
        }
        import os
        torch.save(checkpoint, os.path.join(self.save_path, filename))

    def load_checkpoint(self):
        """
        Load the model checkpoint
        """
        if self.args.train.resume is not None:
            checkpoint = torch.load(self.args.train.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.rt_optimizer.load_state_dict(checkpoint['rt_optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.rt_scheduler.load_state_dict(checkpoint['rt_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['best_epoch']
            pre_val_results = self._eval(mode='val',epoch=checkpoint['epoch'])
            pre_val_results = {**pre_val_results['loss'], **pre_val_results['metric']}
            self.best_val_metrics = pre_val_results[self.args.train.core_metric[0]]
            logger.info(f"Checkpoint loaded at epoch {self.start_epoch}, val metrics {self.args.train.core_metric[0]}:{self.best_val_metrics:.4f}")
        else:
            logger.info("No checkpoint loaded")

    def load_checkpoint_mid(self,path):
        """
        Load the model checkpoint
        """
        if path is not None:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.rt_optimizer.load_state_dict(checkpoint['rt_optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.rt_scheduler.load_state_dict(checkpoint['rt_scheduler'])

    def test(self):
        """
        Test the model
        """
        self.load_checkpoint()
        """
        Evaluate the model
        """
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        self.model.eval()
        
        loader = self.loader['val']
        mode = 'test'
        with torch.no_grad():
            with tqdm(loader, desc=f"Evaluating {mode}", unit='batch') as pbar:
                for _,data in enumerate(pbar):
                    outputs = self.model.eval_step(data,rt=1)
                    losses.update(outputs['loss'].item(), data['image'].size(0))
                    top1.update(outputs['acc1'], data['image'].size(0))
                    top5.update(outputs['acc5'], data['image'].size(0))


        self.model.on_eval_end()
        results = {'loss': {'ce':losses.avg}, 'metric': {'acc1': top1.avg.item(), 'acc5': top5.avg.item()}}