import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
cuda_visible_devices = []#[0]#list(range(8))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cuda_visible_devices)))
import sys
sys.path.append('/home/shlomi.fenster/notebooks/PixelBioID/V2/')
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
import pandas as pd
import pytorch_lightning as pl
from typing import Any, Optional, Type
import timm
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR
import numpy as np
from tqdm.notebook import tqdm
from losses import (NamedWeightedLossClass, SubSetCELoss, 
                    Top1AccForArcFace, Top1AccForSubset, 
                    F1ScoreForSubset, PrecisionForSubset, RecallForSubset,
                    ROCAUCForSubset, WithinSubsetCELoss, OneVsAllCELoss, UNPG)
import random
import json
import pickle
import datetime
from glob import glob
from collections import Counter
from datasets import PixelBioIDGeneralClassDatasetV2, PixelBioIDGeneralClassDatasetMultiVsOtherForDemo
from models import PrefixPlusPretrainedArcFaceModelWithDynamicHPV2, MyStepLRScheduler, ExplicitLRSchedulingScheduler


exp_to_copy = "/mnt/Recordings/outputs/tmp/Experiments/Shlomi/PixelsBioID/18March2024/NewDataJuly2024/exp3"
config = pickle.load(open(os.path.join(exp_to_copy, 'config.pkl'), 'rb'))

embedding_size = config['embedding_size']
learning_rate = config['learning_rate'] / 8
monitor = 'test_SubsetF1'

epochs = 10000

transform_configs = {'train': {'RandomPerspective': 0.1,
                               'RandomRotation': 15,
                               'RandomVerticalFlip': 0.5,
                               'RandomHorizontalFlip': 0.5,
                               'RandomResizedCrop': {'size': (224, 224), 'scale': (0.7, 1.1)},
                               'RandomErasing': 0.2,
                               'ColorJitter': {'brightness': 0.08, 'contrast': 0.08}},
                     'val': {'CenterCrop': (480, 480), 'Resize': (224, 224)}}

train_transforms = []
for k, v in transform_configs['train'].items():
    t_cls = getattr(transforms, k)
    if type(v) is dict:
        t_inst = t_cls(**v)
    else:
        t_inst = t_cls(v)
    train_transforms.append(t_inst)

val_transforms = []
for k, v in transform_configs['val'].items():
    t_cls = getattr(transforms, k)
    if type(v) is dict:
        t_inst = t_cls(**v)
    else:
        t_inst = t_cls(v)
    val_transforms.append(t_inst)

test_transforms = val_transforms


train_df = pd.read_pickle('/mnt/Recordings.SSD/Test/BioID/PixelsBioID/meta_data/V2/oct24_demo_1_train.pkl')
val_df = pd.read_pickle('/mnt/Recordings.SSD/Test/BioID/PixelsBioID/meta_data/V2/oct24_demo_1_val.pkl')
test_df = pd.read_pickle('/mnt/Recordings.SSD/Test/BioID/PixelsBioID/meta_data/V2/oct24_demo_1_test.pkl')

min_num_samples_per_class = 8
batch_size = 64
num_workers = 16

class_column = 'subject'
groupby_column = 'sensor_serial'
class_subset = ['aviad maizels', 'yonatan wexler']
path_column = 'full_png_path'

train_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='train', 
                                           is_train=True, 
                                           meta_data=train_df, 
                                           class_column=class_column,
                                           class_weights={c: 50 for c in class_subset},
                                           groupby_column=groupby_column, 
                                        #    groupby_weights= {'V2_f_2.0_b_2.0_270e010b_NTGM312P00031': 100.},
                                           transforms=train_transforms,
                                           batch_size=batch_size, 
                                           class_subset=class_subset,
                                           min_num_samples_per_class=min_num_samples_per_class,
                                           path_column=path_column,
                                           extra_column_outputs=['sensor_serial',
                                                                 'subject',
                                                                 'selected_power', 
                                                                 'placement_ind', 
                                                                 'wiggle_ind'],
                                           epoch_len=100)
train_dl_kwargs = dict(shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=train_ds.collate_fn)

validation_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='validation', 
                                                is_train=False, 
                                                meta_data=val_df.sample(n=len(val_df)),
                                                class_column=class_column, 
                                                groupby_column=groupby_column, 
                                                transforms=val_transforms,
                                                batch_size=batch_size, 
                                                class_subset=class_subset,
                                                min_num_samples_per_class=min_num_samples_per_class,
                                                path_column=path_column,
                                                extra_column_outputs=['selected_power', 
                                                                      'placement_ind', 
                                                                      'wiggle_ind'])
val_dl_kwargs = dict(batch_size=8, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

test_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='test', 
                                          is_train=False, 
                                          meta_data=test_df, 
                                          class_column=class_column, 
                                          groupby_column=groupby_column, 
                                          transforms=test_transforms,
                                          batch_size=batch_size, 
                                          class_subset=class_subset,
                                          min_num_samples_per_class=min_num_samples_per_class,
                                          path_column=path_column,
                                          extra_column_outputs=['sensor_serial',
                                                                 'subject',
                                                                 'selected_power', 
                                                                 'placement_ind', 
                                                                 'wiggle_ind'])
test_dl_kwargs = dict(batch_size=8, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)


num_classes = len(set(train_df[class_column]))
num_ce_classes = num_classes

unpg = NamedWeightedLossClass(name='UNPG', 
                                      weight=0.,                                       
                                      func=UNPG(num_classes, embedding_size, 
                                                margin=config['unpg']['margin'],
                                                init_centroids=True, train_centroids=False,
                                                label_ind=0,
                                                wisk=config['unpg']['wisk']))

arcface_top1acc_metric = NamedWeightedLossClass(name='ArcFaceAcc',
                                                weight=0.,  
                                                func=Top1AccForArcFace(unpg.func.arcface.weight, label_ind=0), 
                                                is_loss=False)

extra_net = nn.Sequential(nn.Linear(embedding_size, embedding_size // 2), nn.Mish(),
                          nn.Linear(embedding_size // 2, embedding_size // 4), nn.Mish(),
                          nn.Linear(embedding_size // 4, embedding_size // 8), nn.Mish(),
                          nn.Linear(embedding_size // 8, embedding_size // 16), nn.Mish(),
                          nn.Linear(embedding_size // 16, len(class_subset) + 1))
subset_ce = NamedWeightedLossClass(name='SubsetCE',
                       weight=1.,
                       func=SubSetCELoss(extra_net, label_ind=1))

subset_top1acc_metric = NamedWeightedLossClass(name='SubsetAcc',
                                               weight=0.,
                                               func=Top1AccForSubset(subset_ce.func.extra_net, label_ind=1),
                                               is_loss=False)

subset_f1_metric = NamedWeightedLossClass(name='SubsetF1',
                                          weight=0.,
                                          func=F1ScoreForSubset(subset_ce.func.extra_net, label_ind=1),
                                          is_loss=False)

subset_precision_metric = NamedWeightedLossClass(name='SubsetPrecision',
                                                 weight=0.,
                                                 func=PrecisionForSubset(subset_ce.func.extra_net, label_ind=1),
                                                 is_loss=False)

subset_recall_metric = NamedWeightedLossClass(name='SubsetRecall',
                                                 weight=0.,
                                                 func=RecallForSubset(subset_ce.func.extra_net, label_ind=1),
                                                 is_loss=False)

subset_roc_auc_metric = NamedWeightedLossClass(name='SubsetROCAuC',
                                               weight=0.,
                                               func=ROCAUCForSubset(subset_ce.func.extra_net, label_ind=1),
                                               is_loss=False)

within_subset_ce_loss = NamedWeightedLossClass(name='WithinSubsetCE',
                                               weight=1.,
                                               func=WithinSubsetCELoss(subset_ce.func.extra_net, label_ind=1),
                                               is_loss=True)

one_vs_all_ce_loss = NamedWeightedLossClass(name='OneVsAllSubsetCE',
                                            weight=1.,
                                            func=OneVsAllCELoss(subset_ce.func.extra_net, label_ind=1, singled_out_class=0),
                                            is_loss=True)

named_weighted_loss_list = [
    # unpg, arcface_top1acc_metric, 
                            subset_ce, 
                            # subset_top1acc_metric, 
                            subset_f1_metric, 
                            subset_precision_metric,
                            subset_recall_metric,
                            # within_subset_ce_loss, 
                            # one_vs_all_ce_loss, 
                            subset_roc_auc_metric
                            ]

hp_schedulers = []

lr_scheduler_cls = MyStepLRScheduler
lr_scheduler_kwargs = {'lower_bound_lr': 1.64e-4, 'step_size': 80, 'gamma': 0.8}
lr_scheduler_has_monitor = False

val_ds_list=[
    # validation_ds, 
    test_ds
    ]
val_dl_kwargs_list = [
    # val_dl_kwargs, 
    test_dl_kwargs
    ]

model = PrefixPlusPretrainedArcFaceModelWithDynamicHPV2(embedding_size=embedding_size, 
                                                        named_weighted_loss_list=named_weighted_loss_list,
                                                        pretrained_model_name='efficientnet_b2',
                                                        in_channels=1,
                                                        lr=learning_rate, 
                                                        lr_scheduler_cls=lr_scheduler_cls, 
                                                        lr_scheduler_kwargs=lr_scheduler_kwargs, 
                                                        lr_scheduler_has_monitor=lr_scheduler_has_monitor, 
                                                        optimizer_cls=optim.NAdam, 
                                                        hp_schedulers=hp_schedulers,
                                                        train_ds=train_ds,
                                                        train_dl_kwargs=train_dl_kwargs,
                                                        val_ds_list=val_ds_list,
                                                        val_dl_kwargs_list=val_dl_kwargs_list)


exps_dir = "/mnt/Recordings/outputs/tmp/Experiments/Shlomi/PixelsBioID/V2/"
exp_path = Path(exps_dir) / 'demo_sensor__aviad_yonatan_9'

logger = pl.loggers.TensorBoardLogger(exp_path)

ckpt_path = exp_path / "last.ckpt"    
ckpt_path = str(ckpt_path) if ckpt_path.exists() else None

trainer = pl.Trainer(logger=logger, max_epochs=epochs, accelerator='auto', default_root_dir=exp_path, 
                     callbacks=[ModelCheckpoint(dirpath=exp_path, 
                                                monitor=monitor,
                                                mode='max',
                                                save_top_k=50,
                                                save_last=True, 
                                                every_n_epochs=2,
                                                save_on_train_epoch_end=True,
                                                filename='{epoch}-{' + monitor + ':.5f}'),
                                LearningRateMonitor()], 
                     check_val_every_n_epoch=2)#,                     devices=len(cuda_visible_devices))

trainer.fit(model, ckpt_path=ckpt_path)

