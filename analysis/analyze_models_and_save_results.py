import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from data.datasets import PixelBioIDGeneralClassDatasetV2
from model.models import PrefixPlusPretrainedArcFaceModelWithDynamicHPV2
from misc.inference import Inference
from .analysis import BioIDSimulatorAnalyzerV2, EnrolmentMethod
import warnings

tqdm.pandas()
warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)


def run_inference_and_return_path(ckpt_path, test_df_path, recalc=True):
    exp_path = os.path.dirname(ckpt_path)
    ckpt_file_name = os.path.basename(ckpt_path)
    df_pkl_path = os.path.join(exp_path, f'test_inference_df__{ckpt_file_name.split(".ckpt")[0]}_NEW_BLUEPRINT_TEST.pkl')
    
    if os.path.exists(df_pkl_path) and not recalc:
        test_inference_df = pd.read_pickle(df_pkl_path)        
        return test_inference_df, df_pkl_path
    
    
    
    model = PrefixPlusPretrainedArcFaceModelWithDynamicHPV2.load_from_checkpoint(ckpt_path, map_location='cpu')
    model = model.eval()
    model = model.to('cuda')
    
    test_transforms = []
    transforms_dict = {'CenterCrop': (480, 480), 'Resize': (224, 224)}
    for k, v in transforms_dict.items():
        t_cls = getattr(transforms, k)
        if type(v) is dict:
            t_inst = t_cls(**v)
        else:
            t_inst = t_cls(v)
        test_transforms.append(t_inst)

    test_df = pd.read_pickle(test_df_path)
    test_ds = PixelBioIDGeneralClassDatasetV2(dataset_name='test', 
                                              is_train=False, 
                                              meta_data=test_df, 
                                              class_column='subject_', 
                                              transforms=test_transforms,
                                              batch_size=66,   
                                              path_column='local_png_path',
                                              extra_column_outputs=['sensor_serial',
                                                                    'subject_',
                                                                    'selected_power', 
                                                                    'placement_ind', 
                                                                    'wiggle_ind'])
    test_dl_kwargs = model.val_dl_kwargs_list[1]
    test_dl = DataLoader(test_ds, **test_dl_kwargs)
    test_df = test_dl.dataset.meta_data
    
    test_inference = Inference(model, test_dl)
    test_embs, test_labels = test_inference.run()

    tar_id_to_emb = dict()
    for i in range(len(test_df)):
        tar_id_to_emb[test_df.iloc[i]['q_frames_id']] = test_embs[i]

    test_df['embedding'] = test_df['q_frames_id'].map(tar_id_to_emb)
    test_df.sensor_serial = test_df.sensor_serial.apply(lambda x: 'none...' if x is None else x)
    test_inference_df = test_df.copy()
    test_inference_df.to_pickle(df_pkl_path)
    return test_inference_df, df_pkl_path


def run_bio_id_simulator(inference_df_path, recalc=True, enrolment_method='RANDOM',
                         num_enrolment=15, num_wiggles_per_placement=1, num_placements_enrolment=1, num_rep=1):
    all_sign_ins_df_path = inference_df_path.replace('test_inference_df__', 'test_sign_ins_df__')
    if enrolment_method == EnrolmentMethod.WIGGLES_PER_PLACEMENT:
        all_sign_ins_df_path = all_sign_ins_df_path.replace(enrolment_method, enrolment_method + f'_#Wiggles_{num_wiggles_per_placement}')

    if enrolment_method == EnrolmentMethod.CROSS_PLACEMENTS:
        all_sign_ins_df_path = all_sign_ins_df_path.replace(enrolment_method, enrolment_method + f'_#Placements_{num_placements_enrolment}')

    if enrolment_method == EnrolmentMethod.RANDOM:
        all_sign_ins_df_path = all_sign_ins_df_path.replace(enrolment_method, enrolment_method + f'_#Rep_{num_rep}')

    
    if os.path.exists(all_sign_ins_df_path) and not recalc:
        all_sign_ins_df = pd.read_pickle(all_sign_ins_df_path)
    else:
        test_inference_df = pd.read_pickle(inference_df_path)
        ignore_tar_ids = []
        
        bio_id_simulator_analyzer = BioIDSimulatorAnalyzerV2(test_inference_df, min_num_embeddings=30, ignore_tar_ids=ignore_tar_ids, 
                                                             enrolment_method=enrolment_method)
        subject_sensor_pairs = bio_id_simulator_analyzer.get_list_of_subject_sensor_pairs()
        subject_sensor_pairs_for_simulated_users = []
        for su, se in subject_sensor_pairs:
            tmp_df = test_inference_df[(test_inference_df['subject_'] == su) & (test_inference_df['sensor_serial'] == se)]
            num_placements = tmp_df['placement_ind'].nunique()
            if num_placements >= 3:
                subject_sensor_pairs_for_simulated_users.append((su, se))
        all_sign_ins_df, all_enrolments_df = bio_id_simulator_analyzer.get_all_sign_ins_df(subject_sensor_pairs_for_simulated_users, 
                                                                                           num_enrolment=num_enrolment,
                                                                                           num_wiggles_per_placement=num_wiggles_per_placement,
                                                                                           num_placements_enrolment=num_placements_enrolment,
                                                                                           comparison_to_enrolment_func=np.max)
        
        all_sign_ins_df.to_pickle(all_sign_ins_df_path)
        all_enrolments_df.to_pickle(all_sign_ins_df_path.replace('test_sign_ins_df__', 'test_enrolments_df__'))

    return all_sign_ins_df, all_sign_ins_df_path
    

def get_metrics(scores, labels, far=2e-5):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    far_ind = min(range(len(fpr))[::-1], key=lambda i: abs(fpr[i] - far))
    frr = 1 - tpr[far_ind]
    thresh = thresholds[far_ind]
    if np.isinf(thresh):
        thresh = 1.
    return frr, thresh


def get_metrics_from_df(df, far=2e-5):
    scores, labels = df['sign_in_score'].to_numpy(), df['binary_sign_in_label'].to_numpy()
    return get_metrics(scores, labels, far=far)


def get_frr_far(scores, labels, threshold):
    est = (scores >= threshold).astype(int)
    frr = sum(est[labels == 1] == 0) / sum(labels == 1)
    far = sum(est[labels == 0]) / sum(labels == 0)
    return frr, far

def get_far_frr_by_threshold(df, threshold):
    scores, labels = df['sign_in_score'].to_numpy(), df['binary_sign_in_label'].to_numpy()
    return get_frr_far(scores, labels, threshold)


def get_far_frr_func_for_threshold(threshold):
    def get_far_frr_with_fixed_threshold(df):
        return get_far_frr_by_threshold(df, threshold)
    return get_far_frr_with_fixed_threshold

def get_subject_estimation_acc_from_df(df):
    subjects_and_centroids = df.groupby('subject_').embedding.apply(lambda x: np.stack(x).mean(0))
    subjects, centroids = zip(*[(k, v) for k, v in subjects_and_centroids.to_dict().items()])
    centroids = normalize(np.stack(centroids)).T
    y = np.array(df['subject_'].apply(lambda x: subjects.index(x)).to_list())
    emb = normalize(np.stack(df['embedding']))
    class_est = (emb @ centroids).argmax(1)
    acc = (class_est == y).mean()
    return acc

def get_mean_sensor_subject_estimation_acc_from_df(df):
    return df.groupby('sensor_serial').apply(get_subject_estimation_acc_from_df).mean()

    
def run_analysis_and_get_results(exp_path, test_df_path, k_top_models_to_analyze):
    all_results = []

    all_model_paths_to_check = []

    config = pickle.load(open(Path(exp_path) / 'config.pkl', 'rb'))
    monitor = config['monitor']
    model_files = [x for x in os.listdir(exp_path) if 'epoch=' in x and x.endswith('.ckpt')]
    sorted_model_files = sorted(model_files, key=lambda x: float(x.split(f'{monitor}=')[1].split('.ckpt')[0]), reverse=True)
    best_k_models = sorted_model_files[:k_top_models_to_analyze]
    for f in best_k_models:
        all_model_paths_to_check.append(os.path.join(exp_path, f))

    for model_path in tqdm(all_model_paths_to_check):
        test_inference_df, inference_df_path = run_inference_and_return_path(model_path, test_df_path, recalc=False)

        for laser_power in [30, 35, 40, 45, 'All']:
            if laser_power == 'All':
                test_inference_df_specific_laser_power = test_inference_df.copy()
            else:
                test_inference_df_specific_laser_power = test_inference_df[test_inference_df['selected_power'] == laser_power]

            for enrolment_method in EnrolmentMethod._member_names_:
                inference_df_path_specific_laser_power = inference_df_path.replace('.pkl', f'__LaserPower_{laser_power}__Enrolment_{enrolment_method}.pkl')
                test_inference_df_specific_laser_power.to_pickle(inference_df_path_specific_laser_power)
                
                if enrolment_method == EnrolmentMethod.CROSS_PLACEMENTS:
                    for num_placements in [1, 2]:
                        df, df_path = run_bio_id_simulator(inference_df_path_specific_laser_power, recalc=False, 
                                                enrolment_method=enrolment_method, num_placements_enrolment=num_placements)

                        test_acc = get_mean_sensor_subject_estimation_acc_from_df(df)
                        global_thresh_frr, global_thresh = get_metrics_from_df(df)

                        dynamic_frr_and_thresh_dict = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_metrics_from_df).to_dict()
                        dynamic_frrs, dynamic_threshs = zip(*dynamic_frr_and_thresh_dict.values())
                        mean_threshold = np.mean(dynamic_threshs)         
                        fixed_thresh_res = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_far_frr_func_for_threshold(mean_threshold)).to_dict()
                        fixed_thresh_frr_mean, fixed_thresh_far_mean = np.mean(list(zip(*fixed_thresh_res.values())), 1)
                        
                        result = dict(model_path=model_path, test_acc=test_acc, laser_power=laser_power, enrolment_method=enrolment_method, 
                                    num_whatevs=num_placements, mean_dynamic_frr=np.mean(dynamic_frrs), 
                                    global_thresh_frr=global_thresh_frr, global_thresh=global_thresh,
                                    fixed_thresh_frr_mean=fixed_thresh_frr_mean, fixed_thresh_far_mean=fixed_thresh_far_mean,
                                    df_path=df_path)
                        
                        all_results.append(result)
                elif enrolment_method == EnrolmentMethod.WIGGLES_PER_PLACEMENT:
                    for num_wiggles in [1, 2, 3]:
                        df, df_path = run_bio_id_simulator(inference_df_path_specific_laser_power, recalc=False, 
                                                enrolment_method=enrolment_method, num_wiggles_per_placement=num_wiggles)

                        test_acc = get_mean_sensor_subject_estimation_acc_from_df(df)
                        global_thresh_frr, global_thresh = get_metrics_from_df(df)

                        dynamic_frr_and_thresh_dict = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_metrics_from_df).to_dict()
                        dynamic_frrs, dynamic_threshs = zip(*dynamic_frr_and_thresh_dict.values())
                        mean_threshold = np.mean(dynamic_threshs)         
                        fixed_thresh_res = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_far_frr_func_for_threshold(mean_threshold)).to_dict()
                        fixed_thresh_frr_mean, fixed_thresh_far_mean = np.mean(list(zip(*fixed_thresh_res.values())), 1)
                        result = dict(model_path=model_path, test_acc=test_acc, laser_power=laser_power, enrolment_method=enrolment_method, 
                                    num_whatevs=num_wiggles, mean_dynamic_frr=np.mean(dynamic_frrs), 
                                    global_thresh_frr=global_thresh_frr, global_thresh=global_thresh,
                                    fixed_thresh_frr_mean=fixed_thresh_frr_mean, fixed_thresh_far_mean=fixed_thresh_far_mean,
                                    df_path=df_path)
                        all_results.append(result)
                else:
                    for num_rep in [1, 2, 3]:
                        df, df_path = run_bio_id_simulator(inference_df_path_specific_laser_power, recalc=False, 
                                                enrolment_method=enrolment_method, num_rep=num_rep)

                        test_acc = get_mean_sensor_subject_estimation_acc_from_df(df)
                        global_thresh_frr, global_thresh = get_metrics_from_df(df)

                        dynamic_frr_and_thresh_dict = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_metrics_from_df).to_dict()
                        dynamic_frrs, dynamic_threshs = zip(*dynamic_frr_and_thresh_dict.values())
                        mean_threshold = np.mean(dynamic_threshs)         
                        fixed_thresh_res = df.groupby(['sensor_serial', 'bio_id_simulation_user']).apply(get_far_frr_func_for_threshold(mean_threshold)).to_dict()
                        fixed_thresh_frr_mean, fixed_thresh_far_mean = np.mean(list(zip(*fixed_thresh_res.values())), 1)
                        result = dict(model_path=model_path, test_acc=test_acc, laser_power=laser_power, enrolment_method=enrolment_method, 
                                    num_whatevs=num_rep, mean_dynamic_frr=np.mean(dynamic_frrs), 
                                    global_thresh_frr=global_thresh_frr, global_thresh=global_thresh,
                                    fixed_thresh_frr_mean=fixed_thresh_frr_mean, fixed_thresh_far_mean=fixed_thresh_far_mean,
                                    df_path=df_path)
                        all_results.append(result)
                        
    all_results_df = pd.DataFrame(all_results)
    return all_results_df


        