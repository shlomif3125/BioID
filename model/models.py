import pytorch_lightning as pl
from .losses import NamedWeightedLossClass
from typing import Any, Optional
import torch
from torch import nn
from torch import optim
import timm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from copy import copy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR
import functools
from torch.utils.data import DataLoader

import time

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

    
class MyStepLRScheduler(StepLR):
    def __init__(self, optimizer, lower_bound_lr, *args, **kwargs):
        self.optimizer=optimizer
        self._last_lr = [p['lr'] for p in self.optimizer.param_groups]
        self.lower_bound_lr = lower_bound_lr   
        super().__init__(optimizer, *args, **kwargs)
    def step(self):
        curr_lr = self.get_last_lr()[0]
        if curr_lr == self.lower_bound_lr:
            return
        
        if curr_lr * self.gamma <= self.lower_bound_lr:
            new_gamma = self.lower_bound_lr / curr_lr
            self.gamma = new_gamma
        super().step()   
        

class ExplicitLRSchedulingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_to_lr, *args, **kwargs):
        self.optimizer = optimizer
        self.step_to_lr = step_to_lr
        self._last_lr = [p['lr'] for p in self.optimizer.param_groups]
        print(f"INIT LR: {self._last_lr[0]}")
        super().__init__(optimizer, *args, **kwargs)
    
    def get_lr(self):
        if self._step_count in self.step_to_lr.keys():
            print(f"UPDATING LR IN ExpliciLRScheduling: {self.step_to_lr[self._step_count]}")
            return [self.step_to_lr[self._step_count] for p in self.optimizer.param_groups]
        else:
            # print('Not changing lr in explicit thing')
            return self._last_lr
    
    def step(self):
        # print(self._step_count, self._last_lr[0])
        super().step()
    
WD = 0.01

class PrefixPlusPretrainedArcFaceModelWithDynamicHPV2(pl.LightningModule):
    
    def __init__(self, 
                 embedding_size: int = 512,
                 named_weighted_loss_list: list[NamedWeightedLossClass] = [],
                 in_channels: int = 1, 
                 pretrained_model_name: str= 'efficientnet_b0',
                 lr: float = 0.001,
                 lr_scheduler_cls: Optional[any] = None,
                 lr_scheduler_kwargs: dict={},
                 lr_scheduler_has_monitor = True,
                 optimizer_cls = optim.Adam,
                 hp_schedulers: list=[str, dict[tuple[str, ...], dict[int, float]]], 
                 train_ds=None,
                 train_dl_kwargs={},
                 val_ds_list=[],
                 val_dl_kwargs_list=[]
        #         [('ArcFaceLoss', {('loss', 'margin'): {0: np.radians(5), 10000: np.radians(10), 20000: np.radians(20)}}),
        #           ('ArcFaceLoss', {('loss', 'radius'): {0: 3.14, 200: 807231, 2000: -9}}),
        #           ('ArcFaceAcc',  {('abc', 'def'):     {10000: 5e14}})
        #          ]
                 ):
        
        super().__init__()
        print(f"Starting model init...")
        
        self.in_channels = in_channels
        print(f"Creating prefix layers...")
        pre_batch_norm = nn.BatchNorm2d(in_channels)
        prefix_conv = nn.Conv2d(in_channels, 3, 1)
        
        print(f"Loading TIMM model: {pretrained_model_name}")
        pretrained_model = timm.create_model(pretrained_model_name, num_classes=embedding_size, pretrained=True)
        print(f"TIMM model loaded, creating sequential...")
        
        self.model = nn.Sequential(pre_batch_norm, prefix_conv, pretrained_model)
        print(f"Model created, setting up losses...")            
        
        # super().__init__()
        # self.in_channels = in_channels
        # pre_batch_norm = nn.BatchNorm2d(in_channels)
        # prefix_conv = nn.Conv2d(in_channels, 3, 1)
        # pretrained_model = timm.create_model(pretrained_model_name, num_classes=embedding_size, pretrained=True)
        # self.model = nn.Sequential(pre_batch_norm, prefix_conv, pretrained_model)
        
        self.named_weighted_loss_list = named_weighted_loss_list
        self.named_weighted_loss_func_module_dict = nn.ModuleDict({nwl.name: nwl.func 
                                                                   for nwl in named_weighted_loss_list})
        
        train_metrics = {}
        val_metrics = {}
        test_metrics = {}
        for nwl in named_weighted_loss_list:
            if nwl.is_metric:
                l = copy(nwl)
                l.weight = 1.
                train_metrics['train_' + l.name] = self.named_weighted_loss_func_module_dict[l.name]
                val_metrics['val_' + l.name] = self.named_weighted_loss_func_module_dict[l.name]
                # test_metrics['test_' + l.name] = self.named_weighted_loss_func_module_dict[l.name]
        
        print(f"named-weighted-losses compiled")

        
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.lr = lr
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_has_monitor = lr_scheduler_has_monitor
        self.optimizer_cls = optimizer_cls
        self.hp_schedulers = hp_schedulers
        self.train_ds = train_ds
        self.train_dl_kwargs = train_dl_kwargs
        self.val_ds_list = val_ds_list
        self.val_dl_kwargs_list = val_dl_kwargs_list
        
        nwl_name_to_index = {nwl.name: i for i, nwl in enumerate(named_weighted_loss_list)}
        step_to_updates_list = dict()
        for nwl_name, updates_dict in hp_schedulers:
            for update_field, update_steps_and_vals in updates_dict.items():
                # for step, (val, print_log) in update_steps_and_vals.items():
                for step, val_and_print_log in update_steps_and_vals.items():  # BACKWARDS COMPATABILITY
                    try:
                        val, print_log = val_and_print_log
                    except TypeError:
                        val = val_and_print_log
                        print_log = True
                    updates_list = step_to_updates_list.get(step, [])
                    updates_list.append((nwl_name_to_index[nwl_name], update_field, val, print_log))
                    step_to_updates_list[step] = updates_list
                            
        self.sorted_step_to_updates_list = {step: step_to_updates_list[step] for step in sorted(list(step_to_updates_list.keys()))}

        print(f"schedulers set-up complete")

        self.my_global_step = 0

        self.embs_and_meta_data = pd.DataFrame(columns=['subject_', 'sensor_serial', 'embedding'])
        
        self.automatic_optimization = False
        
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['train_ds', 'val_ds_list'])

        
        print(f"Model init complete!")
        
    def forward(self, x):
        return self.model(x)
    
    def log_metrics(self, metrics:dict[str, Any], emb: torch.Tensor, y: torch.Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            for metric_name, metric_func in metrics.items():
                self.log(metric_name, metric_func(emb, y),
                         on_step=False, on_epoch=True, logger=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
    
        
    def update_hps(self, global_step):
        updates_list = self.sorted_step_to_updates_list.get(global_step, [])
        for nwl_ind, field, val, print_log in updates_list:
            curr_val = rgetattr(self.named_weighted_loss_func_module_dict[nwl_ind], '.'.join(field))
            if print_log:
                print(f"GlobalStep {global_step}: Updating {self.named_weighted_loss_list[nwl_ind].name}, {field}: {curr_val} ==> {val}")
            rsetattr(self.named_weighted_loss_func_module_dict[nwl_ind], '.'.join(field), val)
            
                
    def training_step(self, batch, batch_idx):
        self.update_hps(self.my_global_step)
        self.my_global_step += 1

        x, y, y_subset, extra_columns, dataset_name = batch
        ys = torch.stack([y, y_subset], -1).T
        emb = self(x)
        
        loss = 0.
        for nwl in self.named_weighted_loss_list:
            if nwl.is_loss:
                l = self.named_weighted_loss_func_module_dict[nwl.name]
                loss += l(emb, ys) * nwl.weight

        self.log_metrics(self.train_metrics, emb, ys, batch_idx)
        
        opts, lrss = self.configure_optimizers()  
        for i, opt in enumerate(opts):     
            opt.zero_grad()
            if i+1 < len(self.configure_optimizers()[0]):
                opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
                opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
            
        self.manual_backward(loss)
        
        for opt in opts:
            opt.step()        
        
        for lrs in lrss:
            lrs.step()
            
        return loss

    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_ds, **self.train_dl_kwargs)
        return train_dataloader
    
    
    def on_validation_epoch_start(self):
        self.embs_and_meta_data = pd.DataFrame(columns=['subject_', 'sensor_serial', 'embedding', 'ys'])    
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, y_subset, extra_columns, dataset_name = batch
        ys = torch.stack([y, y_subset], -1).T

        dataset_name = dataset_name[0]
        
        if dataset_name == 'validation':
            emb = self(x)
            self.log_metrics(self.val_metrics, emb, ys, batch_idx)
            
        elif dataset_name == 'test':
            with torch.no_grad():
                emb = self(x).cpu().numpy()
            batch_size = emb.shape[0]
            tmp_df = pd.DataFrame([dict(subject_=extra_columns['subject_'][ind], 
                                        sensor_serial=extra_columns['sensor_serial'][ind], 
                                        embedding=emb[ind], ys=ys[:, ind].cpu().numpy()) for ind in range(batch_size)])
            self.embs_and_meta_data = pd.concat([self.embs_and_meta_data, tmp_df])
        else:
            raise ValueError(f"Unrecognized dataset-name {dataset_name}")

    
    @staticmethod
    def get_subject_estimation_acc_from_df(df):
        # print(len(df))
        # print(len(set(df['subject_'])))
        subjects_and_centroids = df.groupby('subject_').embedding.apply(lambda x: np.stack(x).mean(0))
        subjects, centroids = zip(*[(k, v) for k, v in subjects_and_centroids.to_dict().items()])
        centroids = normalize(np.stack(centroids)).T
        y = np.array(df['subject_'].apply(lambda x: subjects.index(x)).to_list())
        emb = normalize(np.stack(df['embedding']))
        class_est = (emb @ centroids).argmax(1)
        acc = (class_est == y).mean()
        return acc

    def on_validation_epoch_end(self): 
        df = self.embs_and_meta_data
        if len(df):
            all_embs = torch.tensor(np.stack(df['embedding'].to_list(), 1), device=self.device).T
            all_ys = torch.tensor(np.stack(df['ys'].to_list(), 1), device=self.device)
            # self.log_metrics(self.test_metrics, all_embs, all_ys, None)
            mean_sensor_subject_estimation_acc = df.groupby('sensor_serial').apply(self.get_subject_estimation_acc_from_df).mean()
            self.log('approx_test_acc', mean_sensor_subject_estimation_acc, sync_dist=True)
        else:
            print('Test df thing empty for some reason...')
        super().on_validation_epoch_end()
        
        
    def val_dataloader(self):
        val_dataloaders = [DataLoader(ds, **dl_kwargs) for ds, dl_kwargs in zip(self.val_ds_list, self.val_dl_kwargs_list)]
        return val_dataloaders
    
    
    def predict_step(self, batch, batch_idx):
        x, *_ = self.unpack_batch(batch)
        emb = self(x)
        return emb
    
    def configure_optimizers(self):        
        base_optimizer = self.optimizer_cls(self.parameters(), lr=self.lr, weight_decay=WD)
        if self.lr_scheduler_cls is None:
            base_scheduler = None
        else:
            base_scheduler = self.lr_scheduler_cls(base_optimizer, **self.lr_scheduler_kwargs)
            if self.lr_scheduler_has_monitor:
                base_scheduler = {"scheduler": base_scheduler,
                                  "monitor": "val_loss",
                                  "frequency": 1}
        
        opts = [base_optimizer]
        lrss = [base_scheduler]
        
        for nwl in self.named_weighted_loss_list:
            if nwl.is_loss:
                nwl_params = [p for p in nwl.func.parameters() if p.requires_grad]
                if len(nwl_params):
                    if nwl.lr_scheduler_config is not None:
                        nwl_lr = nwl.lr_scheduler_config['lr_initial_value']
                    else:
                        nwl_lr = self.lr

                    if nwl.optimizer is not None:
                        opts.append(nwl.optimizer(nwl_params, lr=nwl_lr, weight_decay=WD))
                    else:
                        opts.append(self.optimizer_cls(nwl_params, lr=nwl_lr, weight_decay=WD))

                    if nwl.lr_scheduler_config is not None:
                        lr_class = nwl.lr_scheduler_config['lr_class']
                        lr_scheduler_kwargs = nwl.lr_scheduler_config["lr_scheduler_kwargs"]
                        lr_scheduler_has_monitor = nwl.lr_scheduler_config['lr_scheduler_has_monitor']
                        lrs = lr_class(opts[-1], **lr_scheduler_kwargs)
                        if lr_scheduler_has_monitor:
                            lrs = {"scheduler": lrs,
                                   "monitor": "val_loss",
                                   "frequency": 1}
                        lrss.append(lrs)
                            
        return opts, lrss
    
