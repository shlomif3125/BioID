import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
cuda_visible_devices = [3]#list(range(8))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cuda_visible_devices)))
import sys
sys.path.append('/home/shlomi.fenster/notebooks/PixelBioID/V2/')

from pathlib import Path
from torchvision import transforms
import pandas as pd
from datetime import datetime, timedelta

import pytorch_lightning as pl
# from typing import Any, Optional, Type
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from tqdm.notebook import tqdm
from losses import (NamedWeightedLossClass, Top1AccForArcFace, UNPG)
import pickle
from glob import glob
from datasets import PixelBioIDGeneralClassDatasetV2_TEMP as PixelBioIDGeneralClassDatasetV2
from models import PrefixPlusPretrainedArcFaceModelWithDynamicHPV2, MyStepLRScheduler
from maybe_update_training_data import run_everything_and_return_new_train_path
from analyze_models_and_save_results import run_analysis_and_get_results

exp_to_copy = "/mnt/ML/ModelsTrainResults/shlomi.fenster/PixelsBioID/V2/new_blueprint_data_16Dec2024_split__12Jan2025/"
config = pickle.load(open(os.path.join(exp_to_copy, 'config.pkl'), 'rb'))

config['transform_configs']['train'] = {'RandomPerspective': 0.15,
                                        'RandomRotation': 2,
                                        'RandomVerticalFlip': 0.5,
                                        'RandomHorizontalFlip': 0.5,
                                        'RandomResizedCrop': {'size': (224, 224), 'scale': (0.8, 1.12)},
                                        'RandomErasing': 0.2,
                                        'ColorJitter': {'brightness': 0.1, 'contrast': 0.1}}

exps_dir = "/mnt/ML/ModelsTrainResults/shlomi.fenster/PixelsBioID/V2/"

embedding_size = config['embedding_size']
learning_rate = config['learning_rate'] / 10
monitor = 'approx_test_acc'

epochs = 4000

transform_configs = config['transform_configs']
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


print('GONNA GET ME SOME DATA')
train_df_file = run_everything_and_return_new_train_path(update=False)
print('DATA AQUIRED')

val_df_file = '/mnt/ML/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_val_for_TB_v0.pkl'
test_df_file = '/mnt/ML/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_test_for_TB_v0.pkl'
full_test_df_file = '/mnt/ML/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_test_v0.pkl'
train_df = pd.read_pickle(train_df_file)
val_df = pd.read_pickle(val_df_file)
test_df = pd.read_pickle(test_df_file)

batch_size = 64
min_num_samples_per_class = 22
num_workers = 10

class_column = 'subject_'
path_column = 'local_png_path'

config['learning_rate'] = learning_rate
config['transform_configs'] = transform_configs
config['monitor'] = monitor
config['workers'] = num_workers
config['batch_size'] = batch_size
config['min_num_samples_per_class'] = min_num_samples_per_class
config['train_df_file'] = train_df_file
config['val_df_file'] = val_df_file
config['test_df_file'] = test_df_file
config['epochs'] = epochs
config['exps_dir'] = exps_dir

today_datestr = datetime.strftime(datetime.today(), '%d%b%Y')

exp_path = Path(exps_dir) / f'new_blueprint_data_16Dec2024_split__{today_datestr}_center_crop__efficientnet_b3'
if not exp_path.exists():
    exp_path.mkdir(parents=True)
pickle.dump(config, open(exp_path / 'config.pkl', 'wb'))


train_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='train',
                                           is_train=True, 
                                           meta_data=train_df, 
                                           class_column=class_column,
                                           target_crop_size = (140, 200),
                                        #    subsample_factor = (3, 3),
                                           transforms=train_transforms,
                                           # min_num_samples_per_class=min_num_samples_per_class, 
                                           batch_size=batch_size, 
                                        #    class_subset=class_subset,
                                           path_column=path_column,
                                           extra_column_outputs=['sensor_serial',
                                                                 'subject_',
                                                                 'selected_power', 
                                                                 'placement_ind', 
                                                                 'wiggle_ind'],
                                           epoch_len=100)
train_dl_kwargs = dict(shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True, collate_fn=train_ds.collate_fn)

validation_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='validation', 
                                                is_train=False, 
                                                meta_data=val_df.sample(n=len(val_df)),
                                                class_column=class_column, 
                                                target_crop_size = (140, 200),
                                                # subsample_factor = (3, 3),
                                                transforms=val_transforms,
                                                batch_size=batch_size, 
                                                # class_subset=class_subset,
                                                path_column=path_column,
                                                extra_column_outputs=['selected_power', 
                                                                      'placement_ind', 
                                                                      'wiggle_ind'])
val_dl_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=False)

test_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='test', 
                                          is_train=False, 
                                          meta_data=test_df.sample(n=len(test_df)), 
                                          class_column=class_column, 
                                          target_crop_size = (140, 200),
                                        #   subsample_factor = (3, 3),
                                          transforms=test_transforms,
                                          batch_size=batch_size, 
                                        #   class_subset=class_subset,
                                          path_column=path_column,
                                          extra_column_outputs=['sensor_serial',
                                                                 'subject_',
                                                                 'selected_power', 
                                                                 'placement_ind', 
                                                                 'wiggle_ind'])
test_dl_kwargs = dict(batch_size=int(batch_size * 1.5), shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=False)


num_classes = len(set(train_df[class_column]))
num_ce_classes = num_classes

unpg = NamedWeightedLossClass(name='UNPG', 
                              weight=1.,     
                              func=UNPG(num_classes, embedding_size, 
                                                margin=config['unpg']['margin'],
                                                init_centroids=True, train_centroids=False,
                                                label_ind=0,
                                                wisk=config['unpg']['wisk']))

arcface_top1acc_metric = NamedWeightedLossClass(name='ArcFaceAcc',
                                                weight=0.,  
                                                func=Top1AccForArcFace(unpg.func.arcface.weight, 
                                                                       label_ind=0), 
                                                is_loss=False)


named_weighted_loss_list = [
                            unpg, 
                            arcface_top1acc_metric,
                            ]

hp_schedulers = []

lr_scheduler_cls = MyStepLRScheduler
lr_scheduler_kwargs = {'lower_bound_lr': 1.64e-4, 'step_size': 80, 'gamma': 0.8}
lr_scheduler_has_monitor = False

val_ds_list=[
    validation_ds, 
    test_ds
    ]
val_dl_kwargs_list = [
    val_dl_kwargs, 
    test_dl_kwargs
    ]

print('============================================')
print('GONNA GET ME A MODEL INITIALIZED')
model = PrefixPlusPretrainedArcFaceModelWithDynamicHPV2(embedding_size=embedding_size, 
                                                        named_weighted_loss_list=named_weighted_loss_list,
                                                        pretrained_model_name='efficientnet_b3',
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
print('MODEL AQUIRED')


logger = pl.loggers.TensorBoardLogger(exp_path)

ckpt_path = exp_path / "last.ckpt"    
ckpt_path = str(ckpt_path) if ckpt_path.exists() else None

print('GONNA TRAIN ME A MODEL')
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
                     check_val_every_n_epoch=2,
                     detect_anomaly=False)#,                     devices=len(cuda_visible_devices))

trainer.fit(model, ckpt_path=ckpt_path)
print('MODEL TRAINED')

print('GONNA GET ME SOME ANALYSIS')
analysis_results = run_analysis_and_get_results(exp_path, full_test_df_file, 5)
print('ANALYSIS AQUIRED')

analysis_results.to_pickle(exp_path / 'analysis_results.pkl')
print('ABSOLUTELY AND COMPLETELY DONE DUDE')

