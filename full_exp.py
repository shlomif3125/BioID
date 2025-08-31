import os
import pickle
from pathlib import Path
import pandas as pd
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torchvision import transforms

from model.losses import (NamedWeightedLossClass, Top1AccForArcFace, UNPG)
from data.datasets import PixelBioIDGeneralClassDatasetV2_TEMP as PixelBioIDGeneralClassDatasetV2
from model.models import PrefixPlusPretrainedArcFaceModelWithDynamicHPV2, MyStepLRScheduler
from data.maybe_update_training_data import run_everything_and_return_new_train_path
from analysis.analyze_models_and_save_results import run_analysis_and_get_results
from utils import transforms_list_from_dict

import hydra

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    exps_dir = config.trainer.exps_dir
    embedding_size = config.model.emebdding_size
    monitor = config.trainer.monitor
    epochs = config.trainer.epochs

    learning_rate = config.model.learning_rate

    train_transforms = transforms_list_from_dict(config.data.train.transforms)
    val_transforms = transforms_list_from_dict(config.data.val.transforms)
    test_transforms = transforms_list_from_dict(config.data.test.transforms)

    if config.data.train.update_train_data:
        print('GONNA GET ME SOME DATA')
        train_df_file = run_everything_and_return_new_train_path()
        print('DATA AQUIRED')
    else:
        train_df_file = config.data.train.df_file

    train_df = pd.read_pickle(train_df_file)
    val_df = pd.read_pickle(config.data.val.df_file)
    test_df = pd.read_pickle(config.data.test.df_file)

    today_datestr = datetime.strftime(datetime.today(), '%d%b%Y')

    exp_path = Path(exps_dir) / f'{config.trainer.exp_name}__{today_datestr}'
    if not exp_path.exists():
        exp_path.mkdir(parents=True)
    pickle.dump(config, open(exp_path / 'config.pkl', 'wb'))


    train_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='train',
                                            is_train=True, 
                                            meta_data=train_df, 
                                            class_column=class_column,
                                            transforms=train_transforms,
                                            batch_size=batch_size, 
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
                                                    transforms=val_transforms,
                                                    batch_size=batch_size, 
                                                    path_column=path_column,
                                                    extra_column_outputs=['selected_power', 
                                                                        'placement_ind', 
                                                                        'wiggle_ind'])
    val_dl_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=False)

    test_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='test', 
                                            is_train=False, 
                                            meta_data=test_df.sample(n=len(test_df)), 
                                            class_column=class_column, 
                                            transforms=test_transforms,
                                            batch_size=batch_size, 
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
                                                    init_centroids=True, train_centroids=True,
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
                                                            pretrained_model_name=config.model.pretrained_model_name,
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


if __name__ == '__main__':
    main()