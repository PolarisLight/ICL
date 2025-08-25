"""
This file defines the Submodulelr_Trainer class, which is a subclass of Default_Trainer.
The Submodulelr_Trainer class is used to train the model with different learning rates for different submodules.
It is useful when we have different modules in the model and some of them have pretrained weights.
"""
import torch
from loguru import logger
from Trainer.default_trainer import Default_Trainer

class Submodulelr_Trainer(Default_Trainer):
    def __init__(self,args,model,dataset,device):
        super().__init__(args,model,dataset,device)
        try:
            params_list = [{"params": getattr(model, name).parameters(), "lr": param} for name, param in vars(args.submodule).items()]
            self.optimizer = getattr(torch.optim, self.args.train.optimizer)(params_list, weight_decay=self.args.train.weight_decay)
            logger.bind(params=True).info(f"="*50)
            logger.bind(params=True).info(f"Using submodule learning rates: ")
            for name, param in vars(args.submodule).items():
                logger.bind(params=True).info(f"  {name}: {param:.2e}")
            logger.bind(params=True).info(f"="*50)
        except AttributeError:
            raise ValueError(f"Unknown optimizer: {self.args.train.optimizer}")
        
    def adjust_learning_rate(self, epoch, lr):
        if self.scheduler is None:
            return
        for i, (name, param) in enumerate(vars(self.args.submodule).items()):
            new_lr = self.scheduler.calculate_lr(param, epoch)
            self.optimizer.param_groups[i]['lr'] = new_lr
            if new_lr != param and self.args.scheduler.name == 'step':
                logger.info(f"[Epoch {epoch}] Update submodule {name} learning rate from {param:.2e} to {new_lr:.2e}")