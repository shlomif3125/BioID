import argparse
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'    


def main():
    import sys
    sys.path.append('/home/shlomi.fenster/notebooks/PixelBioID/March18/')
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()
    import pickle
    from pathlib import Path
    import pickle
    from glob import glob
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch import optim
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt
    import random
    import warnings
    from losses import Top1AccForArcFace
    from filtering_data_for_analysis.misc import get_metrics_from_df, get_far_frr_func_for_threshold, get_video_im, get_list_of_video_ims, show_specific_row
    from datasets import PixelsBioIDDataset
    from models import PrefixPlusPretrainedArcFaceModel, MyStepLRScheduler, PrefixPlusPretrainedArcFaceModelWithDynamicHPV2
    from inference import Inference
    from analysis import BioIDSimulatorAnalyzerV2
    from filtering_data_for_analysis.subjects_and_sensors import is_bad_sensor, is_ignore_subject_sensor_pair
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import hashlib
    from sklearn.preprocessing import normalize


    test_df_path = '/mnt/Recordings.SSD/Test/BioID/PixelsBioID/meta_data/01082023_13042024_V0_test_NoAnnoyingStuff_v3.pkl'
    test_df = pd.read_pickle(test_df_path)
    test_df['local_png_path'] = test_df['local_png_path'].apply(lambda x: x.replace('NVM.GPU04', 'NVM'))


    class FixedWrappedDataset(PixelsBioIDDataset):
        def __getitem__(self, idx):
            x, ys, *_ = super().__getitem__(idx)
            return x, ys

    def run_inference_and_return_path(ckpt_path, recalc=True):
        exp_path = os.path.dirname(ckpt_path)
        ckpt_file_name = os.path.basename(ckpt_path)
        df_pkl_path = os.path.join(exp_path, f'test_inference_df__{ckpt_file_name.split(".ckpt")[0]}.pkl')

        if os.path.exists(df_pkl_path) and not recalc:
            test_inference_df = pd.read_pickle(df_pkl_path)        
            return test_inference_df, df_pkl_path

        batch_size = 32
        workers = 20
        learning_rate = 1.  # not used, yuck
        embedding_size = 512

        crop_size = (672, 672)
        resize_size = (224, 224)


        test_subjects = sorted(list(set(test_df.subject)))
        test_subject2idx = {s: i for i, s in enumerate(test_subjects)}

        test_transforms = (transforms.CenterCrop(crop_size), 
                           transforms.Resize(resize_size))
        test_ds = FixedWrappedDataset('test', test_df, test_subject2idx, test_transforms)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, 
                                     pin_memory=True)   

        model = PrefixPlusPretrainedArcFaceModelWithDynamicHPV2.load_from_checkpoint(ckpt_path)

        test_inference = Inference(model, test_dataloader)
        test_preds, test_ys = test_inference.run()
        assert np.all(test_ys[:, 0] == test_ys[:, 1])
        test_embs = test_preds.copy()
        test_labels = test_ys[:, 0].astype(np.float32)

        tar_id_to_emb = dict()
        for i in tqdm(range(len(test_df))):
            tar_id_to_emb[test_df.iloc[i].tar_id] = test_embs[i]

        test_df['embedding'] = test_df.tar_id.map(tar_id_to_emb)
        test_df.sensor_serial = test_df.sensor_serial.apply(lambda x: 'none...' if x is None else x)
        test_inference_df = test_df.copy()
        test_inference_df.to_pickle(df_pkl_path)
        return test_inference_df, df_pkl_path

    def run_bio_id_simulator(inference_df_path, recalc=True):
        sign_ins_df_path = inference_df_path.replace('test_inference_df__', 'test_sign_ins_df__')
        if os.path.exists(sign_ins_df_path) and not recalc:
            all_sign_ins_df = pd.read_pickle(sign_ins_df_path)
        else:
            test_inference_df = pd.read_pickle(inference_df_path)
            remove_these = test_inference_df.progress_apply(lambda r: is_bad_sensor(r) or is_ignore_subject_sensor_pair(r) or r.subject == 'unknown unknown' , axis=1)
            ignore_tar_ids = test_inference_df[remove_these].tar_id.to_list()
            bio_id_simulator_analyzer = BioIDSimulatorAnalyzerV2(test_inference_df, min_num_embeddings=30, ignore_tar_ids=ignore_tar_ids)
            all_sign_ins_df, all_enrolments_df = bio_id_simulator_analyzer.get_all_sign_ins_df(num_enrolment=15, comparison_to_enrolment_func=np.max)

            all_sign_ins_df.to_pickle(inference_df_path.replace('test_inference_df__', 'test_sign_ins_df__'))
            all_enrolments_df.to_pickle(inference_df_path.replace('test_inference_df__', 'test_enrolments_df__'))

        return all_sign_ins_df


    def get_subject_estimation_acc_from_df(df):
        subjects_and_centroids = df.groupby('subject').embedding.apply(lambda x: np.stack(x).mean(0))
        subjects, centroids = zip(*[(k, v) for k, v in subjects_and_centroids.to_dict().items()])
        centroids = normalize(np.stack(centroids)).T
        y = np.array(df['subject'].apply(lambda x: subjects.index(x)).to_list())
        emb = normalize(np.stack(df['embedding']))
        class_est = (emb @ centroids).argmax(1)
        acc = (class_est == y).mean()
        return acc

    def get_mean_sensor_subject_estimation_acc_from_df(df):
        return df.groupby('sensor_serial').apply(get_subject_estimation_acc_from_df).mean()


    def get_last_val_acc(ckpt_path):
        path = os.path.dirname(glob(os.path.join(os.path.dirname(ckpt_path), 'lightning_logs', 'version*', 'events*'))[0])
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
        return runlog_data[runlog_data['metric'] == 'val_ArcFaceAcc/dataloader_idx_0'].sort_values('step').iloc[-1]['value']



    results_dir = '../continued_exps/short_models_analysis_outputs'
    # if not os.path.exists(results_dir):
    #     os.mkdir(results_dir)

    warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
    
    ckpts_to_run = glob('/mnt/Recordings/outputs/tmp/Experiments/Shlomi/PixelsBioID/18March2024/NewDataJuly2024/exp7/epoch*.ckpt')
    random.shuffle(ckpts_to_run)

    for ckpt_num, ckpt_path in enumerate(ckpts_to_run):  
        pseudo_out_path = os.path.join(results_dir, hashlib.sha256(ckpt_path.encode('utf-8')).hexdigest() + '.tmp')
        if os.path.exists(pseudo_out_path):
            continue

        with open(pseudo_out_path, 'w'):
            pass

        out_path = os.path.join(results_dir, hashlib.sha256(ckpt_path.encode('utf-8')).hexdigest() + '.pkl')
        if os.path.exists(out_path):
            continue

        res = dict()
        print(f"{ckpt_path}, ({ckpt_num}/{len(ckpts_to_run)})")
        test_inference_df, inference_df_path = run_inference_and_return_path(ckpt_path, recalc=False)

        exp = ckpt_path.split('/')[ckpt_path.split('/').index('NewDataJuly2024') + 1]
        epoch = torch.load(ckpt_path)['epoch']

        config = pickle.load(open(os.path.join(os.path.dirname(ckpt_path), 'config.pkl'), 'rb'))
        unpg_config = config['unpg']
        unpg_wisk = unpg_config['wisk']
        unpg_margin = unpg_config['margin']


        df = run_bio_id_simulator(inference_df_path, recalc=False)

        test_acc = get_mean_sensor_subject_estimation_acc_from_df(df)
        if os.path.basename(ckpt_path) == 'last.ckpt':
            val_tb_acc = get_last_val_acc(ckpt_path)
        else:
            val_tb_acc = float(os.path.basename(ckpt_path).split('=')[-1].split('.ckpt')[0])

        global_thresh_frr, global_thresh = get_metrics_from_df(df)

        dynamic_frr_and_thresh_dict = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_metrics_from_df).to_dict()
        dynamic_frrs, dynamic_threshs = zip(*dynamic_frr_and_thresh_dict.values())

        mean_threshold = np.mean(dynamic_threshs)         
        fixed_thresh_res = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_far_frr_func_for_threshold(mean_threshold)).to_dict()
        fixed_thresh_frr_mean, fixed_thresh_far_mean = np.mean(list(zip(*fixed_thresh_res.values())), 1)

        res['exp'] = exp
        res['epoch'] = epoch
        res['unpg_wisk'] = unpg_wisk
        res['unpg_margin'] = unpg_margin
        res['test_acc'] = test_acc
        res['val_tb_acc'] = val_tb_acc
        res['fixed_thresh_frr_mean'] = fixed_thresh_frr_mean
        res['fixed_thresh_far_mean'] = fixed_thresh_far_mean
        res['fixed_thresh_mean'] = mean_threshold
        res['dynamic_thresh_frr'] = np.mean(dynamic_frrs)
        res['global_thresh_frr'] = global_thresh_frr
        res['global_thresh'] = global_thresh
        pickle.dump(res, open(out_path, 'wb'))
        os.remove(pseudo_out_path)



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu')
    args = parser.parse_args()
    args_gpu = args.gpu
    if args_gpu is None:
        args_gpu = ','.join(list(map(str, list(range(8)))))      
    os.environ['CUDA_VISIBLE_DEVICES'] = args_gpu
    
    main()

