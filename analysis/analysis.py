import numpy as np
from itertools import product
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import pandas as pd
import random
from sklearn.metrics import roc_curve, roc_auc_score
from enum import Enum


class EnrolmentMethod(str, Enum):
    RANDOM = 'RANDOM'
    CROSS_PLACEMENTS = 'CROSS_PLACEMENTS'
    WIGGLES_PER_PLACEMENT = 'WIGGLES_PER_PLACEMENT'
    
    
class BioIDSimulatorAnalyzerV2:
    def __init__(self, inference_df: pd.DataFrame, min_num_embeddings: int=20, ignore_tar_ids: list=[], 
                 enrolment_method: EnrolmentMethod=EnrolmentMethod.RANDOM) -> None:
        if len(ignore_tar_ids):
            inference_df = inference_df[~inference_df['q_frames_id'].isin(ignore_tar_ids)]
        self.inference_df = inference_df
        self.min_num_embeddings = min_num_embeddings
        self.enrolment_method = enrolment_method
    
    def simple_sign_in_and_impostor_attempts(self, subject, sensor_serial, num_enrolment=5, use_seed=True, 
                                             comparison_to_enrolment_func=np.mean):
        # Per-Sensor-Subject
        # Score calculated as mean-distance from enrolment embeddings
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_df = inference_df[(inference_df['subject_'] == subject) & 
                                             (inference_df.sensor_serial == sensor_serial)]
        subject_and_sensor_embeddings = np.stack(subject_and_sensor_df.embedding.to_list())

        all_impostors_df = inference_df[(inference_df['subject_'] != subject) & 
                                        (inference_df.sensor_serial == sensor_serial)]
        all_impostor_sign_in_embeddings = all_impostors_df.embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None

        num_samples = subject_and_sensor_embeddings.shape[0]
        user_enrolment_inds = random.sample(list(range(num_samples)), num_enrolment)
        user_sign_in_inds = [i for i in range(num_samples) if i not in user_enrolment_inds]
        user_enrolment_embeddings = subject_and_sensor_embeddings[user_enrolment_inds]
        user_sign_in_embeddings = subject_and_sensor_embeddings[user_sign_in_inds]
        
        user_enrolment_df = subject_and_sensor_df.iloc[user_enrolment_inds]
        user_sign_ins_df = subject_and_sensor_df.iloc[user_sign_in_inds]

        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        sign_in_attempts = np.concatenate([user_sign_in_embeddings, impostor_sign_in_embeddings])

        scores = (1 + comparison_to_enrolment_func(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)) / 2
        
        sign_ins_df = pd.concat([user_sign_ins_df, all_impostors_df])
        sign_ins_df['bio_id_simulation_user'] = subject
        sign_ins_df['sign_in_score'] = scores
        sign_ins_df['binary_sign_in_label'] = labels
      
        sign_ins_df['closest_user_enrolment_tar_id'] = user_enrolment_df.iloc[np.argmax(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)]['q_frames_id'].to_list()
     
        user_enrolment_df['bio_id_simulation_user'] = subject

        return sign_ins_df, user_enrolment_df    
    
    
    def cross_placements_enrolment_sign_in_and_impostor_attempts(self, subject, sensor_serial, num_placements_enrolment=1, use_seed=True, 
                                                                 comparison_to_enrolment_func=np.mean):
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_df = inference_df[(inference_df['subject_'] == subject) & 
                                             (inference_df.sensor_serial == sensor_serial)]
        placements = sorted(set(subject_and_sensor_df['placement_ind']))
        enrolment_placements = random.sample(placements, num_placements_enrolment)
        user_enrolment_df = subject_and_sensor_df[subject_and_sensor_df['placement_ind'].isin(enrolment_placements)]
        user_enrolment_embeddings = np.stack(user_enrolment_df.embedding.to_list())
        
        user_sign_ins_df = subject_and_sensor_df[~subject_and_sensor_df['placement_ind'].isin(enrolment_placements)]
        user_sign_in_embeddings = np.stack(user_sign_ins_df.embedding.to_list())

        all_impostors_df = inference_df[(inference_df['subject_'] != subject) & 
                                        (inference_df.sensor_serial == sensor_serial)]
        all_impostor_sign_in_embeddings = all_impostors_df.embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None
        
        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        sign_in_attempts = np.concatenate([user_sign_in_embeddings, impostor_sign_in_embeddings])

        scores = (1 + comparison_to_enrolment_func(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)) / 2
        
        sign_ins_df = pd.concat([user_sign_ins_df, all_impostors_df])
        sign_ins_df['bio_id_simulation_user'] = subject
        sign_ins_df['sign_in_score'] = scores
        sign_ins_df['binary_sign_in_label'] = labels
      
        sign_ins_df['closest_user_enrolment_tar_id'] = user_enrolment_df.iloc[np.argmax(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)]['q_frames_id'].to_list()
     
        user_enrolment_df['bio_id_simulation_user'] = subject

        return sign_ins_df, user_enrolment_df
    
    
    def single_placement_enrolment_sign_in_and_impostor_attempts(self, subject, sensor_serial, use_seed=True, 
                                             comparison_to_enrolment_func=np.mean):
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_df = inference_df[(inference_df['subject_'] == subject) & 
                                             (inference_df.sensor_serial == sensor_serial)]
        placements = sorted(set(subject_and_sensor_df['placement_ind']))
        enrolment_placement = random.choice(placements)
        user_enrolment_df = subject_and_sensor_df[subject_and_sensor_df['placement_ind'] == enrolment_placement]
        user_enrolment_embeddings = np.stack(user_enrolment_df.embedding.to_list())
        
        user_sign_ins_df = subject_and_sensor_df[subject_and_sensor_df['placement_ind'] != enrolment_placement]
        user_sign_in_embeddings = np.stack(user_sign_ins_df.embedding.to_list())

        all_impostors_df = inference_df[(inference_df['subject_'] != subject) & 
                                        (inference_df.sensor_serial == sensor_serial)]
        all_impostor_sign_in_embeddings = all_impostors_df.embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None
        
        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        sign_in_attempts = np.concatenate([user_sign_in_embeddings, impostor_sign_in_embeddings])

        scores = (1 + comparison_to_enrolment_func(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)) / 2
        
        sign_ins_df = pd.concat([user_sign_ins_df, all_impostors_df])
        sign_ins_df['bio_id_simulation_user'] = subject
        sign_ins_df['sign_in_score'] = scores
        sign_ins_df['binary_sign_in_label'] = labels
      
        sign_ins_df['closest_user_enrolment_tar_id'] = user_enrolment_df.iloc[np.argmax(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)]['q_frames_id'].to_list()
     
        user_enrolment_df['bio_id_simulation_user'] = subject

        return sign_ins_df, user_enrolment_df
    
    def wiggles_from_each_placement_enrolment_sign_in_and_impostor_attempts(self, subject, sensor_serial, num_wiggles_per_placement=1, use_seed=True, 
                                                                           comparison_to_enrolment_func=np.mean):
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_df = inference_df[(inference_df['subject_'] == subject) & 
                                             (inference_df.sensor_serial == sensor_serial)]
        
        placement_to_wiggles_dict = subject_and_sensor_df.groupby('placement_ind')[['placement_ind', 'wiggle_ind']].sample(num_wiggles_per_placement).groupby('placement_ind')['wiggle_ind'].apply(list).to_dict()
        user_enrolment_df = subject_and_sensor_df[subject_and_sensor_df.apply(lambda r: r['wiggle_ind'] in placement_to_wiggles_dict[r['placement_ind']], axis=1)]
        user_enrolment_embeddings = np.stack(user_enrolment_df.embedding.to_list())
        
        user_sign_ins_df = subject_and_sensor_df[subject_and_sensor_df.apply(lambda r: r['wiggle_ind'] not in placement_to_wiggles_dict[r['placement_ind']], axis=1)]
        user_sign_in_embeddings = np.stack(user_sign_ins_df.embedding.to_list())

        all_impostors_df = inference_df[(inference_df['subject_'] != subject) & 
                                        (inference_df.sensor_serial == sensor_serial)]
        all_impostor_sign_in_embeddings = all_impostors_df.embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None
        
        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        sign_in_attempts = np.concatenate([user_sign_in_embeddings, impostor_sign_in_embeddings])

        scores = (1 + comparison_to_enrolment_func(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)) / 2
        
        sign_ins_df = pd.concat([user_sign_ins_df, all_impostors_df])
        sign_ins_df['bio_id_simulation_user'] = subject
        sign_ins_df['sign_in_score'] = scores
        sign_ins_df['binary_sign_in_label'] = labels
      
        sign_ins_df['closest_user_enrolment_tar_id'] = user_enrolment_df.iloc[np.argmax(user_enrolment_embeddings @ sign_in_attempts.T, axis=0)]['q_frames_id'].to_list()
     
        user_enrolment_df['bio_id_simulation_user'] = subject

        return sign_ins_df, user_enrolment_df
    
    def get_list_of_subject_sensor_pairs(self, min_num_embeddings=None):
        if min_num_embeddings is None:
            min_num_embeddings = self.min_num_embeddings
         
        subject_and_sensor_counts = self.inference_df[['subject_', 'sensor_serial']].value_counts()
        subject_sensor_pairs = subject_and_sensor_counts[subject_and_sensor_counts >= min_num_embeddings].index.to_list()
        return subject_sensor_pairs
            

    def get_all_sign_ins_df(self, subject_sensor_pairs=None, ignore_subject_sensor_pairs=None, 
                            num_enrolment=5, num_wiggles_per_placement=1, num_placements_enrolment=1,
                            comparison_to_enrolment_func=np.mean,
                            num_attempts=1, attempts_agg_func=None):        
        if subject_sensor_pairs is None:
            subject_sensor_pairs = self.get_list_of_subject_sensor_pairs()
        
        subject_sensor_pairs = list(map(tuple, subject_sensor_pairs))        
        if ignore_subject_sensor_pairs is not None:
            ignore_subject_sensor_pairs = list(map(tuple, ignore_subject_sensor_pairs))
            subject_sensor_pairs = [x for x in subject_sensor_pairs if x not in ignore_subject_sensor_pairs]
        
        
        all_sign_ins_df = None
        all_enrolments_df = None
        pbar = subject_sensor_pairs
        # pbar = tqdm(subject_sensor_pairs)
        for subject, sensor_serial in pbar:
            if attempts_agg_func is None:
                match self.enrolment_method:
                    case EnrolmentMethod.RANDOM:
                        sign_ins_df, enrolments_df = self.simple_sign_in_and_impostor_attempts(subject, sensor_serial, num_enrolment=num_enrolment,
                                                                                           comparison_to_enrolment_func=comparison_to_enrolment_func)
                    # case EnrolmentMethod.SINGLE_PLACEMENT:
                    #     sign_ins_df, enrolments_df = self.single_placement_enrolment_sign_in_and_impostor_attempts(subject, sensor_serial,                                                                                                                    comparison_to_enrolment_func=comparison_to_enrolment_func)

                    case EnrolmentMethod.CROSS_PLACEMENTS:
                        sign_ins_df, enrolments_df = self.cross_placements_enrolment_sign_in_and_impostor_attempts(subject, sensor_serial, 
                                                                                                                   num_placements_enrolment=num_placements_enrolment,                                                                                                                   comparison_to_enrolment_func=comparison_to_enrolment_func)
                    case EnrolmentMethod.WIGGLES_PER_PLACEMENT:
                        sign_ins_df, enrolments_df = self.wiggles_from_each_placement_enrolment_sign_in_and_impostor_attempts(subject, sensor_serial,                                                                                                                              num_wiggles_per_placement=num_wiggles_per_placement,                                                                                                                              comparison_to_enrolment_func=comparison_to_enrolment_func)
                    case _:
                        print(self.enrolment_method)
                    
            else:
                raise NotImplementedError('Maybe use the old version...')
         
            if sign_ins_df is None:
                continue
            
            if all_sign_ins_df is None:
                all_sign_ins_df = sign_ins_df.copy()
                all_enrolments_df = enrolments_df.copy()
            else:
                all_sign_ins_df = pd.concat([all_sign_ins_df, sign_ins_df])
                all_enrolments_df = pd.concat([all_enrolments_df, enrolments_df])
            
        return all_sign_ins_df, all_enrolments_df       
     


class IntraAndInterSubjectMeanCosineSimilarityMetric:
    def __call__(self, embs, labels):
        return self.intra_and_inter_subject_mean_cosine_sims(embs, labels)

    @staticmethod
    def intra_and_inter_subject_mean_cosine_sims(embs, labels):
        y_r = np.tile(labels.reshape(-1, 1), len(labels))
        y_c = y_r.T
        triu = np.triu(np.ones((len(labels), len(labels))), 1)
        intra_subject = (y_r == y_c) * triu
        inter_subject = (y_r != y_c) * triu
        all_cos = 0.5 * ((embs @ embs.T) + 1)
        intra_subject_cos = all_cos * intra_subject
        inter_subject_cos = all_cos * inter_subject
        mean_intra_subject_cos = intra_subject_cos.sum() / intra_subject.sum()
        mean_inter_subject_cos = inter_subject_cos.sum() / inter_subject.sum()
        return {
            "intra": mean_intra_subject_cos,
            "inter": mean_inter_subject_cos,
            "intra_inter_llr": np.log(mean_intra_subject_cos / mean_inter_subject_cos),
        }

    
    
class PairwiseSimilarityRadiusAnalyzer:
    def __init__(self, embs, labels):
        self.embs = embs
        self.labels = labels
        self.results = None
    
    @staticmethod
    def radius_neighbors_results(embs, labels, thresh, min_neighbors):
        cos_sim_mat = embs@embs.T
        thresh_masked_labels = np.stack([labels] * len(labels), 1)
        thresh_masked_labels[np.eye(len(labels)) == 1] = np.nan
        thresh_masked_labels[cos_sim_mat < thresh] = np.nan
        mode, count = stats.mode(thresh_masked_labels, nan_policy='omit', keepdims=False)
        enough_neighbors_mask = count >= min_neighbors
        acc = ((mode == labels)[enough_neighbors_mask]).mean()
        percent_data = enough_neighbors_mask.mean()
        return acc, percent_data
    
    
    def calc_radius_neighbors_grid_results(self, similarity_thresholds=np.arange(0.5, 1., 0.1), min_neighbors_thresholds=range(1, 12, 2)):
        grid = list(product(similarity_thresholds, min_neighbors_thresholds))
        results = []
        for similarity_thresh, min_neighbors_thresh in tqdm(grid):
            acc, percent_data = self.radius_neighbors_results(self.embs, self.labels, similarity_thresh, min_neighbors_thresh)
            results.append((similarity_thresh, min_neighbors_thresh, acc, percent_data))
        
        self.results = results
        return results
    
    def get_working_point(self, acc_value=None, percent_data_value=None):
        assert (acc_value is None) ^ (percent_data_value is None), "You must choose a target accuracy value XOR a target percent-data value"
        if self.results is None:
            print('Grid results have not been calculated. Using default grid values')
            results = self.calc_radius_neighbors_grid_results()
        else:
            results = self.results
        
        if acc_value is not None:
            return sorted(results, key=lambda res: abs(res[2] - acc_value))[0]
        
        if percent_data_value is not None:
            return sorted(results, key=lambda res: abs(res[3] - percent_data_value))[0]
            


class PairwiseSimilarityKNNAnalyzer:
    def __init__(self, embs, labels):
        self.embs = embs
        self.labels = labels
        self.results = None
        self.dists_and_neighbors = None
        
    def create_dists_and_neighbors(self, n_neighbors):
        if self.dists_and_neighbors is None:
            print('Creating NN...')
            nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='cosine')
            nn = nn.fit(self.embs)
            dists, neighbs = nn.kneighbors(self.embs)
            self.dists_and_neighbors = (dists[:, 1:], neighbs[:, 1:])
            print('Done creating NN!')
            
        
    def knn_neighbors_results(self, at_least_percent_consensus, num_neighbors):
        dists, neighbs = self.dists_and_neighbors
        dists = 1 - dists[:, :num_neighbors]
        neighbs = neighbs[:, :num_neighbors]
        neighb_identities = self.labels[neighbs]
        neighbs_mode = stats.mode(neighb_identities, 1, keepdims=False)
        strong_consensus = neighbs_mode.count > np.round(num_neighbors * at_least_percent_consensus)
        acc = np.nanmean((neighbs_mode.mode == self.labels)[strong_consensus])
        percent_data = strong_consensus.mean()
        return acc, percent_data
    
    
    def calc_knn_grid_results(self, at_least_percent_consensus_thresholds=np.arange(0.3, 1., 0.1), num_knn_neighbours=range(1, 18)):
        self.create_dists_and_neighbors(max(num_knn_neighbours))
        
        grid = list(product(at_least_percent_consensus_thresholds, num_knn_neighbours))
        results = []
        for at_least_percent_consensus, num_neighbors in tqdm(grid):
            acc, percent_data = self.knn_neighbors_results(at_least_percent_consensus, num_neighbors)
            results.append((at_least_percent_consensus, num_neighbors, acc, percent_data))
        
        self.results = results
        return results
    
    def get_working_point(self, acc_value=None, percent_data_value=None):
        assert (acc_value is None) ^ (percent_data_value is None), "You must choose a target accuracy value XOR a target percent-data value"
        if self.results is None:
            print('Grid results have not been calculated. Using default grid values')
            results = self.calc_knn_grid_results()
        else:
            results = self.results
        
        if acc_value is not None:
            return sorted(results, key=lambda res: abs(res[2] - acc_value))[0]
        
        if percent_data_value is not None:
            return sorted(results, key=lambda res: abs(res[3] - percent_data_value))[0]
            

    
class BioIDSimulatorAnalyzer:
    def __init__(self, inference_df: pd.DataFrame, min_num_embeddings: int=20) -> None:
        self.inference_df = inference_df
        self.min_num_embeddings = min_num_embeddings
    
    def simple_sign_in_and_impostor_attempts(self, subject, sensor_serial, num_enrolment=5, use_seed=True, 
                                             comparison_to_enrolment_func=np.mean):
        # Per-Sensor-Subject
        # Score calculated as mean-distance from enrolment embeddings
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_df = inference_df[(inference_df['subject_'] == subject) & 
                                             (inference_df.sensor_serial == sensor_serial)]
        subject_and_sensor_embeddings = np.stack(subject_and_sensor_df.embedding.to_list())

        all_impostors_df = inference_df[(inference_df['subject_'] != subject) & 
                                        (inference_df.sensor_serial == sensor_serial)]
        all_impostor_sign_in_embeddings = all_impostors_df.embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None, None

        num_samples = subject_and_sensor_embeddings.shape[0]
        user_enrolment_inds = random.sample(list(range(num_samples)), num_enrolment)
        user_sign_in_inds = [i for i in range(num_samples) if i not in user_enrolment_inds]
        user_enrolment_embeddings = subject_and_sensor_embeddings[user_enrolment_inds]
        user_sign_in_embeddings = subject_and_sensor_embeddings[user_sign_in_inds]
        
        user_enrolment_df = subject_and_sensor_df.iloc[user_enrolment_inds]
        user_sign_ins_df = subject_and_sensor_df.iloc[user_sign_in_inds]

        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        all_sign_in_attempts = np.concatenate([user_sign_in_embeddings, impostor_sign_in_embeddings])

        scores = (1 + comparison_to_enrolment_func(user_enrolment_embeddings @ all_sign_in_attempts.T, axis=0)) / 2
        
        all_sign_ins_df = pd.concat([user_sign_ins_df, all_impostors_df])
        return scores, labels, all_sign_ins_df
    
    def simple_sign_in_and_impostor_with_multiple_attempts(self, subject, sensor_serial, num_enrolment=5, use_seed=True,
                                                           comparison_to_enrolment_func=np.max,
                                                           num_attempts=1, attempts_agg_func=np.mean):
        inference_df = self.inference_df
        if use_seed:
            random.seed(124)
        subject_and_sensor_embeddings = np.stack(inference_df[(inference_df['subject_'] == subject) & 
                                                         (inference_df.sensor_serial == sensor_serial)].embedding.to_list())

        all_impostor_sign_in_embeddings = inference_df[(inference_df['subject_'] != subject) & 
                                                      (inference_df.sensor_serial == sensor_serial)].embedding.to_list()
        
        if len(all_impostor_sign_in_embeddings) < self.min_num_embeddings:
            return None, None

        num_samples = subject_and_sensor_embeddings.shape[0]
        user_enrolment_inds = random.sample(list(range(num_samples)), num_enrolment)
        user_sign_in_inds = [i for i in range(num_samples) if i not in user_enrolment_inds]
        user_enrolment_embeddings = subject_and_sensor_embeddings[user_enrolment_inds]
        user_sign_in_embeddings = subject_and_sensor_embeddings[user_sign_in_inds]

        impostor_sign_in_embeddings = np.stack(all_impostor_sign_in_embeddings)       
        
        
        impostor_labels = inference_df[(inference_df['subject_'] != subject) & 
                                       (inference_df.sensor_serial == sensor_serial)]['subject_'].to_list()
        impostor_label_to_inds = {subj: list(np.where(np.array(impostor_labels) == subj)[0]) for subj in set(impostor_labels)}

        impostor_inds_list_by_labels = [impostor_label_to_inds[subj] for subj in impostor_labels]
        impostor_multiple_attempts_inds = np.array([random.sample(x, num_attempts) for x in impostor_inds_list_by_labels])
        impostor_multiple_attempts_embeddings = impostor_sign_in_embeddings[impostor_multiple_attempts_inds]

        num_user_sign_ins = user_sign_in_embeddings.shape[0]
        user_multiple_attempts_inds = np.array([random.sample(list(range(num_user_sign_ins)), num_attempts) for _ in range(num_user_sign_ins)])
        user_multiple_attempts_embeddings = user_sign_in_embeddings[user_multiple_attempts_inds]

        all_sign_in_attempts = np.concatenate([user_multiple_attempts_embeddings, impostor_multiple_attempts_embeddings])
        multiple_attempt_scores = (1 + comparison_to_enrolment_func(np.einsum('ij,jkl->ikl',user_enrolment_embeddings,all_sign_in_attempts.T), axis=0)) / 2
        
        labels = np.array([1] * user_sign_in_embeddings.shape[0] + [0] * impostor_sign_in_embeddings.shape[0])
        scores = attempts_agg_func(multiple_attempt_scores, 0)
        
        return scores, labels
    
    
    def get_list_of_subject_sensor_pairs(self, min_num_embeddings=None):
        if min_num_embeddings is None:
            min_num_embeddings = self.min_num_embeddings
         
        subject_and_sensor_counts = self.inference_df[['subject_', 'sensor_serial']].value_counts()
        subject_sensor_pairs = subject_and_sensor_counts[subject_and_sensor_counts >= min_num_embeddings].index.to_list()
        return subject_sensor_pairs
            
    @staticmethod
    def get_stats(scores, labels):
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        if len(set(labels)) == 1:
            auc = -1
        else:
            auc = roc_auc_score(labels, scores)

        return fpr, tpr, thresholds, auc, labels.sum(), (1 - labels).sum()

    def run_full_stats(self, subject_sensor_pairs=None, ignore_subject_sensor_pairs=None, num_enrolment=5,
                       comparison_to_enrolment_func=np.mean,
                       num_attempts=1, attempts_agg_func=None):        
        if subject_sensor_pairs is None:
            subject_sensor_pairs = self.get_list_of_subject_sensor_pairs()
        
        subject_sensor_pairs = list(map(tuple, subject_sensor_pairs))        
        if ignore_subject_sensor_pairs is not None:
            ignore_subject_sensor_pairs = list(map(tuple, ignore_subject_sensor_pairs))
            subject_sensor_pairs = [x for x in subject_sensor_pairs if x not in ignore_subject_sensor_pairs]
        
        subject_and_sensor_to_performance = dict()
        pbar = subject_sensor_pairs
        # pbar = tqdm(subject_sensor_pairs)
        for subject, sensor_serial in pbar:
            if attempts_agg_func is None:
                scores, labels, _ = self.simple_sign_in_and_impostor_attempts(subject, sensor_serial, num_enrolment=num_enrolment,
                                                                           comparison_to_enrolment_func=comparison_to_enrolment_func)
            else:
                scores, labels = self.simple_sign_in_and_impostor_with_multiple_attempts(subject, sensor_serial, num_enrolment=num_enrolment,
                                                                                         comparison_to_enrolment_func=comparison_to_enrolment_func,
                                                                                         num_attempts=num_attempts, attempts_agg_func=attempts_agg_func)
                
            if scores is None:
                fpr = tpr = thresholds = auc = num_pos = num_neg = None
            else:
                fpr, tpr, thresholds, auc, num_pos, num_neg = self.get_stats(scores, labels)
            
            if auc is not None and auc < 0:
                print(subject, sensor_serial, 'single class')
                
            if fpr is not None:
                subject_and_sensor_to_performance[subject, sensor_serial] = dict(fpr=fpr, tpr=tpr, 
                                                                                 thresholds=thresholds, auc=auc, 
                                                                                 num_pos=num_pos, num_neg=num_neg)

        subject_and_sensor_results_df = pd.DataFrame([{'subject_': subject, 'sensor': sensor, **v} for (subject, sensor), v in subject_and_sensor_to_performance.items()])
        subject_and_sensor_results_df['neg_to_pos_ratio'] = subject_and_sensor_results_df['num_neg'] / subject_and_sensor_results_df['num_pos']

        for far_target_exp in range(1, 5):
            subject_and_sensor_results_df[f'far1e-{far_target_exp}_ind'] = subject_and_sensor_results_df['fpr'].apply(lambda x: min(range(len(x))[::-1], key=lambda i: abs(x[i] - 1 / (10 ** far_target_exp))))
            subject_and_sensor_results_df[f'frr@far1e-{far_target_exp}'] = subject_and_sensor_results_df.apply(lambda r: 1 - r['tpr'][r[f'far1e-{far_target_exp}_ind']], axis=1)
        
        return subject_and_sensor_results_df
        
        
        
    