from os import path as osp
import pandas as pd
import warnings
import random
import numpy as np
from tqdm import tqdm
import re
tqdm.pandas()
from datetime import datetime, timedelta
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from itertools import product
from pathlib import Path
import shutil
from glob import glob
import multiprocessing as mp

from q_commons.database_tools.db_clients.postgresql_client import PostgreSQLClient, DEFAULT_CONFIG_PATH
from q_commons.database_tools.db_clients.q_metadata_db_views import Q_FRAMES_DETAILS_VIEW_COLUMNS, Q_FRAMES_DETAILS_VIEW_NAME


date_str = datetime.strftime(datetime.today() - timedelta(days=1), '%d%b%Y')

def camel_case_split(full_name):
    return ' '.join(list(map(lambda x: x.lower(), re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', full_name))))

def add_wiggle_ind(experiment_and_sensor_df):
    wiggle_ind = 0
    # experiment_and_sensor_df = experiment_and_sensor_df[experiment_and_sensor_df['text'] = 'Hit spacebar'].sort_values('stage_index')
    
    for i, r in experiment_and_sensor_df.iterrows():
        if r['text'] == 'Hit spacebar':
            experiment_and_sensor_df.loc[i, 'wiggle_ind'] = wiggle_ind
        else:
            if r['text'] == 'Take 10 seconds to remember how the headset feels on your head':
                pass
            elif r['text'] == 'Slightly wiggle each earpiece. Make sure it fits comfortably. Hit spacebar to continue.':
                wiggle_ind += 1
            else:
                raise ValueError('Unexpected text...')
    
    res_df = experiment_and_sensor_df[experiment_and_sensor_df['text'] == 'Hit spacebar']
    res_df = res_df.drop(['session_id', 'sensor_serial'], axis='columns')
    res_df['wiggle_ind'] = res_df['wiggle_ind'].astype(int)
    
    return res_df


def add_placement_ind(subject_sensor_df):
    subject_sensor_df = subject_sensor_df.sort_values('stage_index')
    exp_to_placement_ind = {k: i for i, k in enumerate(subject_sensor_df['session_id'].drop_duplicates().to_list())}
    subject_sensor_df['placement_ind'] = subject_sensor_df['session_id'].map(exp_to_placement_ind)
    return subject_sensor_df


def run_everything_and_return_new_train_path(update=True):
    dfs_dir = '/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/'

    prev_data_df_files = list(map(str, Path(dfs_dir).glob('all_data_till_*.pkl')))
    latest_prev_data_df_file = max(prev_data_df_files, key=lambda x: datetime.strptime(x.split('all_data_till_')[1].split('.pkl')[0], '%d%b%Y'))
    if not update:
        return latest_prev_data_df_file
    latest_prev_data_df = pd.read_pickle(latest_prev_data_df_file)
    latest_prev_date_str = datetime.strftime(latest_prev_data_df['recording_date'].max() - timedelta(days=1), '%d%b%Y')

    columns = ['q_frames_id',
            'q_frames_filename',
            'stage_dir',
            'text',
            'stage_index',
            'session_id',
            'recording_date',
            'flow_name',
            'sensor_serial',
            'placement',
            'fps',
            'selected_power']

    select_columns_string = ', '.join(['qfd.' + col for col in columns])

    print('Running queries...')

    client = PostgreSQLClient.load_from_ini(DEFAULT_CONFIG_PATH)

    new_bio_id_blueprint_id = 'b5188515-91a3-4062-81e3-3b1c74e483ab'

    query_by_bio_id_blueprint_id = f"select {select_columns_string} from " \
                                    "q_frames_details qfd " \
                                f"where qfd.blueprint_id = '{new_bio_id_blueprint_id}' " \
                                    "and qfd.is_cancelled = False " \
                                    "and qfd.is_skipped = False " \
                                f"and qfd.recording_date >= '{str(latest_prev_date_str)}'"

    new_bio_id_blueprint_df = client.load_into_df(query_by_bio_id_blueprint_id, columns)

    print('Done')

    new_bio_id_blueprint_df['recording_date'] = new_bio_id_blueprint_df['recording_date'].apply(lambda x: x.date())
    new_bio_id_blueprint_df = new_bio_id_blueprint_df[new_bio_id_blueprint_df['fps'] == 200.]


    new_blueprint_exp_id_stage_nums = new_bio_id_blueprint_df['session_id'].value_counts()
    new_blueprint_completed_exps = new_blueprint_exp_id_stage_nums[new_blueprint_exp_id_stage_nums == 78].index.to_list()
    completed_exps_new_blueprint_df = new_bio_id_blueprint_df[new_bio_id_blueprint_df['session_id'].isin(new_blueprint_completed_exps)]

    completed_exps_new_blueprint_df['subject_'] = completed_exps_new_blueprint_df['stage_dir'].apply(lambda x: x.split('/')[-2].split('-')[0]).progress_apply(camel_case_split)
    completed_exps_new_blueprint_df = completed_exps_new_blueprint_df[completed_exps_new_blueprint_df['subject_'].apply(lambda x: len(set(x.split())) == 2)]


    new_blueprint_with_wiggles_df = completed_exps_new_blueprint_df.groupby(['session_id', 'sensor_serial']).apply(add_wiggle_ind)
    new_blueprint_with_wiggles_df = new_blueprint_with_wiggles_df.reset_index().drop(columns='level_2')

    new_blueprint_with_wiggles_and_placement_df = new_blueprint_with_wiggles_df.groupby(['subject_', 'sensor_serial']).apply(add_placement_ind)
    new_blueprint_data_df = new_blueprint_with_wiggles_and_placement_df.droplevel([0, 1]).reset_index().drop(columns='index')

    new_blueprint_data_df['full_png_path'] = new_blueprint_data_df['stage_dir'] + '/' + new_blueprint_data_df['q_frames_filename'].apply(lambda x: x.replace('.zip', '.png'))
    print(len(set(new_blueprint_data_df['subject_'])))

    new_blueprint_data_df['has_data'] = new_blueprint_data_df['full_png_path'].progress_apply(lambda x: osp.exists(x))
    new_blueprint_data_df = new_blueprint_data_df[new_blueprint_data_df['has_data']].drop(columns=['has_data'])
    print(len(set(new_blueprint_data_df['subject_'])))

    local_storage_path = Path('/mnt/NVM/shlomi/V2BioIDData/')
    remote_storage_path = '/mnt/A3000/Recordings/v2_data/'
    remote_storage_path_new = '/mnt/A3000/Recordings/laser/'
    if not local_storage_path.exists():
        local_storage_path.mkdir(parents=True)
        
    def full_png_path_to_local_path(full_png_path):
        if full_png_path.startswith(remote_storage_path):
            file_path_within_storage = full_png_path.split(remote_storage_path)[1]
        elif full_png_path.startswith(remote_storage_path_new):
            file_path_within_storage = full_png_path.split(remote_storage_path_new)[1]
        else:
            print(full_png_path)
            raise FileNotFoundError()
        local_path = local_storage_path / file_path_within_storage
        return local_path

    def save_to_local_png(full_png_path, override=False):
        save_to = full_png_path_to_local_path(full_png_path)
        if not override and save_to.exists():
            return
        
        save_to_dir = save_to.parent
        if not save_to_dir.exists():
            save_to_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(full_png_path, save_to.as_posix())
        
    def save_list_of_pngs(list_of_full_png_paths, override=False, process_ind=0):
        if process_ind == 0:
            list_of_full_png_paths = tqdm(list_of_full_png_paths)
        
        for full_png_path in list_of_full_png_paths:
            save_to_local_png(full_png_path, override)


    all_full_png_paths = new_blueprint_data_df['full_png_path'].to_list()
    random.shuffle(all_full_png_paths)

    num_procs = 90

    procs = [mp.Process(target=save_list_of_pngs, args=(all_full_png_paths[k::num_procs], False, k)) for k in range(num_procs)]
    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print('Done!')

    new_blueprint_data_df['local_png_path'] = new_blueprint_data_df['full_png_path'].progress_apply(full_png_path_to_local_path).apply(Path.as_posix)
    new_blueprint_data_df = new_blueprint_data_df[new_blueprint_data_df['local_png_path'].progress_apply(osp.exists)]
    new_blueprint_data_df = new_blueprint_data_df[latest_prev_data_df.columns]

    all_data_df = pd.concat([latest_prev_data_df, new_blueprint_data_df])
    all_data_df = all_data_df.drop_duplicates('q_frames_id')

    all_data_df.groupby('flow_name')['subject_'].apply(lambda x: len(set(x)))

    print(date_str)

    all_data_df.to_pickle(osp.join(dfs_dir, f'all_data_till_{date_str}.pkl'))

    new_bio_id_data = all_data_df[all_data_df['flow_name'] == 'positioning_1x8']


    NEW_SPLIT = False  # TODO: only works as "FALSE"...

    if NEW_SPLIT:
        num_sessions_per_subj_sens = new_bio_id_data.groupby(['subject_', 'sensor_serial'])['session_id'].nunique()
        at_least_3_sessions_subj_sens_pairs = num_sessions_per_subj_sens[num_sessions_per_subj_sens >= 3].reset_index()[['subject_', 'sensor_serial']]

        at_least_3__subjs = sorted(set(at_least_3_sessions_subj_sens_pairs['subject_']))
        one_hundred_subjs_for_test = random.sample(at_least_3__subjs, 60)
        test_subj_sens_pairs = at_least_3_sessions_subj_sens_pairs[at_least_3_sessions_subj_sens_pairs['subject_'].isin(one_hundred_subjs_for_test)]
        main_test_data_df = new_bio_id_data.merge(test_subj_sens_pairs, how='inner', on=['subject_', 'sensor_serial'])      

        less_than_3_sessions_subj_sens_pairs = num_sessions_per_subj_sens[num_sessions_per_subj_sens < 3].reset_index()[['subject_', 'sensor_serial']]
        extra_test_data_df = new_bio_id_data.merge(less_than_3_sessions_subj_sens_pairs, how='inner', on=['subject_', 'sensor_serial'])      

        test_df = pd.concat([main_test_data_df, extra_test_data_df])

        train_val_df = new_bio_id_data[~new_bio_id_data['q_frames_id'].isin(test_data_df['q_frames_id'])]

        val_df = train_val_df.groupby(['subject_', 'sensor_serial', 'placement_ind']).sample(n=3)
        train_df = train_val_df[~train_val_df['q_frames_id'].isin(val_df['q_frames_id'].to_list())]


        len(val_df) / len(train_df), len(test_df) / len(train_df)

        test_df.to_pickle(f'/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_{datestr}_test_v0.pkl')
        val_df.to_pickle(f'/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_{datestr}_val_v0.pkl')
        train_df.to_pickle(f'/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_{datestr}_train_v0.pkl')



    def num_subjects_with_at_least_3_exps(dataframe):
        subj_to_num_exps = dataframe.groupby('subject_')['session_id'].apply(lambda x: len(set(x)))
        return (subj_to_num_exps >= 3).sum()


    num_subjects_per_day_list = []
    for rd in sorted(set(new_bio_id_data['recording_date'])):
        num_subjects_per_day = {'Date': rd, 
                                '# Unique Participants': len(set(new_bio_id_data[new_bio_id_data['recording_date'] == rd]['subject_'])), 
                                '# Participants With At Least 3 Exps': num_subjects_with_at_least_3_exps(new_bio_id_data[new_bio_id_data['recording_date'] == rd])}
        num_subjects_per_day_list.append(num_subjects_per_day)
        
    num_subjects_per_day_df = pd.DataFrame(num_subjects_per_day_list)

    print(num_subjects_per_day_df)

    cur_test_df = pd.read_pickle('/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_test_v0.pkl')
    cur_val_df = pd.read_pickle('/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_val_v0.pkl')
    cur_train_paths_format = '/mnt/ModelsTrainResults/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_train_v'
    cur_train_df_path = f'{cur_train_paths_format}0.pkl'
    cur_train_df = pd.read_pickle(cur_train_df_path)

    train_df_paths = glob(f'{cur_train_paths_format}*.pkl')
    versions = [int(f.split(cur_train_paths_format)[1].split('.pkl')[0]) for f in train_df_paths]

    new_version = max(versions) + 1

    cur_test_participants = sorted(set(cur_test_df['subject_']))

    cur_train_tar_ids = cur_train_df['q_frames_id'].to_list()
    cur_val_tar_ids = cur_val_df['q_frames_id'].to_list()

    cur_train_and_val_ids = cur_train_tar_ids + cur_val_tar_ids

    new_bio_id_data_no_subjects_from_test = new_bio_id_data[~new_bio_id_data['subject_'].isin(cur_test_participants)]
    potential_new_train_data = new_bio_id_data_no_subjects_from_test[~new_bio_id_data_no_subjects_from_test['q_frames_id'].isin(cur_train_and_val_ids)]

    potential_participants_to_num_exps_per_sensor = potential_new_train_data.groupby(['subject_', 'sensor_serial'])['session_id'].nunique()
    new_train_subj_sens_pairs = potential_participants_to_num_exps_per_sensor[potential_participants_to_num_exps_per_sensor >= 3].reset_index()[['subject_', 'sensor_serial']]

    new_train_data = potential_new_train_data.merge(new_train_subj_sens_pairs, on=['subject_', 'sensor_serial'])

    new_train_df = pd.concat([cur_train_df, new_train_data])

    new_train_df_path = f'{cur_train_paths_format}{new_version}.pkl'
    print(new_train_df_path)
    new_train_df.to_pickle(new_train_df_path)
    return new_train_df_path