from pathlib import Path
import pandas as pd
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim

from model.losses import (NamedWeightedLossClass, Top1AccForArcFace, UNPG)
from data.datasets import PixelBioIDGeneralClassDatasetV2_TEMP as PixelBioIDGeneralClassDatasetV2
from model.models import PrefixPlusPretrainedArcFaceModelWithDynamicHPV2, MyStepLRScheduler
from data.maybe_update_training_data import run_everything_and_return_new_train_path
from data.utils import get_ds_via_cfg
from analysis.analyze_models_and_save_results import run_analysis_and_get_results

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    lr_cfg = model_cfg.lr_config
    trainer_cfg = cfg.trainer
    analysis_cfg = cfg.analysis
    
    exp_path = Path(trainer_cfg.exps_dir) / f'{trainer_cfg.exp_name}'
    if not exp_path.exists():
        exp_path.mkdir(parents=True)
        
    with open(exp_path /"full_config.yaml", "w") as config_file:
        config_file.write(OmegaConf.to_yaml(cfg))

    if 'train' in cfg.pipeline_stages:

        if data_cfg.train.update_train_data:
            print('GONNA GET ME SOME DATA')
            train_df_file = run_everything_and_return_new_train_path()
            print('DATA AQUIRED')
        else:
            train_df_file = data_cfg.train.df_file

        print('GONNA GET ME SOME DATASETS')
        train_df = pd.read_pickle(train_df_file)
        train_ds = get_ds_via_cfg(data_cfg.train, train_df)
        train_dl_kwargs = dict(shuffle=True, num_workers=data_cfg.num_workers, 
                        pin_memory=False, drop_last=True, collate_fn=train_ds.collate_fn)
        
        val_df = pd.read_pickle(data_cfg.val.df_file)
        validation_ds = get_ds_via_cfg(data_cfg.val, val_df)
        val_dl_kwargs = dict(batch_size=data_cfg.batch_size, shuffle=False, 
                        num_workers=data_cfg.num_workers, pin_memory=False, drop_last=False)
        
        test_df = pd.read_pickle(data_cfg.test.df_file)
        test_ds = get_ds_via_cfg(data_cfg.test, test_df)
        test_dl_kwargs = dict(batch_size=int(data_cfg.batch_size * 1.5), shuffle=False, 
                        num_workers=data_cfg.num_workers, pin_memory=False, drop_last=False)
        print('DONE GETTING DATASETS')

        num_classes = len(set(train_df[data_cfg.class_column]))

        unpg = NamedWeightedLossClass(name='UNPG', 
                                    weight=1.,     
                                    func=UNPG(num_classes, model_cfg.embedding_size, 
                                                        margin=model_cfg.unpg.margin,
                                                        init_centroids=model_cfg.init_centroids, 
                                                        train_centroids=model_cfg.train_centroids,
                                                        label_ind=0,
                                                        wisk=model_cfg.unpg.wisk),)

        arcface_top1acc_metric = NamedWeightedLossClass(name='ArcFaceAcc',
                                                        weight=0.,  
                                                        func=Top1AccForArcFace(unpg.func.arcface.weight, 
                                                                            label_ind=0), 
                                                        is_loss=False)

        named_weighted_loss_list = [unpg, arcface_top1acc_metric]

        hp_schedulers = []

        lr_scheduler_cls = MyStepLRScheduler
        lr_scheduler_kwargs_dict = OmegaConf.to_container(lr_cfg.lr_scheduler_kwargs, resolve=True)

        
        val_ds_list=[validation_ds, test_ds]
        val_dl_kwargs_list = [val_dl_kwargs, test_dl_kwargs]

        print('============================================')
        print('GONNA GET ME A MODEL INITIALIZED')
        model = PrefixPlusPretrainedArcFaceModelWithDynamicHPV2(embedding_size=model_cfg.embedding_size, 
                                                                named_weighted_loss_list=named_weighted_loss_list,
                                                                pretrained_model_name=model_cfg.pretrained_model_name,
                                                                in_channels=1,
                                                                lr=lr_cfg.learning_rate, 
                                                                lr_scheduler_cls=lr_scheduler_cls, 
                                                                lr_scheduler_kwargs=lr_scheduler_kwargs_dict, 
                                                                lr_scheduler_has_monitor=lr_cfg.lr_scheduler_has_monitor, 
                                                                optimizer_cls=optim.NAdam, 
                                                                hp_schedulers=hp_schedulers,
                                                                train_ds=train_ds,
                                                                train_dl_kwargs=train_dl_kwargs,
                                                                val_ds_list=val_ds_list,
                                                                val_dl_kwargs_list=val_dl_kwargs_list)
        print('MODEL AQUIRED')

        # import yaml

        # try:
        #     yaml.dump(model.hparams)
        #     print("✅ YAML dump succeeded on hparams")
        # except Exception as e:
        #     print("YAML dump failed on hparams:", e)
        #     for k, v in model.hparams.items():
        #         try:
        #             yaml.dump({k: v})
        #         except Exception as e2:
        #             print(f"❌ Key {k} with value {v!r} failed: {e2}")

        logger = pl.loggers.TensorBoardLogger(exp_path)

        ckpt_path = exp_path / "last.ckpt"    
        ckpt_path = str(ckpt_path) if ckpt_path.exists() else None

        print('GONNA TRAIN ME A MODEL')
        trainer = pl.Trainer(logger=logger, 
                             max_epochs=trainer_cfg.epochs, 
                             accelerator='auto', 
                             default_root_dir=exp_path,
                             callbacks=[ModelCheckpoint(dirpath=exp_path, 
                                                        monitor=trainer_cfg.monitor,
                                                        mode=trainer_cfg.monitor_mode,
                                                        save_top_k=50,
                                                        save_last=True, 
                                                        every_n_epochs=2,
                                                        save_on_train_epoch_end=True,
                                                        filename='{epoch}-{' + trainer_cfg.monitor + ':.5f}'),
                                        LearningRateMonitor()], 
                             check_val_every_n_epoch=2,
                             detect_anomaly=False)

        trainer.fit(model, ckpt_path=ckpt_path)
        print('MODEL TRAINED')

    if 'analysis' in cfg.pipeline_stages:
        print('GONNA GET ME SOME ANALYSIS')
        analysis_results = run_analysis_and_get_results(exp_path, analysis_cfg.dataset, analysis_cfg.k_top_models_to_analyze)
        print('ANALYSIS AQUIRED')

        analysis_results.to_pickle(exp_path / 'analysis_results.pkl')
        print('ANALYSIS SAVED')

    print('ABSOLUTELY AND COMPLETELY DONE DUDE')


if __name__ == '__main__':
    main()