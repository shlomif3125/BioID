import os.path as osp
import numpy as np
from pathlib import Path
import pandas as pd
import imageio.v2 as imageio
import torch
from torch import nn
from torch.utils.data import Dataset
import random
from typing import Optional


IND_SUFFIX = '_ind'
SUBSET_SUFFIX = '_subset'


class PixelBioIDGeneralClassDatasetV2(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 is_train: bool,
                 meta_data: pd.DataFrame,
                 class_column: str,
                 batch_size: int,
                 class_subset: Optional[list[str]] = None,
                 class_weights: Optional[dict[str, float]] = None,
                 groupby_column: Optional[str] = None,
                 groupby_weights: Optional[dict[str, float]] = None,
                 target_crop_size: Optional[tuple[int, int]] = None,
                 subsample_factor: Optional[tuple[int, int]] = None,
                 transforms: tuple[nn.Module, ...] = tuple(),
                 min_num_samples_per_class: int = 1,
                 sample_with_replacement: bool = False,
                 path_column: str = 'local_png_path',
                 create_npy_files: bool = False,
                 extra_column_outputs: list[str] = [],  # TODO: allow for string-valued columns
                 epoch_len: Optional[int] = 0,
                 ) -> None:
        super().__init__()
        
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.meta_data = meta_data
        self.class_column = class_column
        self.batch_size = batch_size
        self.class_subset = class_subset if class_subset is not None else sorted(set(meta_data[class_column]))
        
        self.preprocess_class_columns()

        self.class_weights_dict = {k: 1. for k in set(meta_data[self.class_column])}
        if class_weights is not None:
            self.class_weights_dict.update(class_weights)        
        
        self.groupby_column = groupby_column
        if groupby_column is not None:
            self.groupby_vals = sorted(set(self.meta_data[self.groupby_column]))
        
            groupby_weights_dict = {k: 1. for k in self.groupby_vals}
            if groupby_weights is not None:
                groupby_weights_dict.update(groupby_weights)
            self.groupby_weights = [groupby_weights_dict[k] for k in self.groupby_vals]
        else:
            self.groupby_vals = None
            self.groupby_weights = None
        
        self.target_crop_size = target_crop_size
        self.subsample_factor = subsample_factor
        self.transforms = nn.Sequential(*transforms)
        self.min_num_samples_per_class = min_num_samples_per_class
        self.sample_with_replacement = sample_with_replacement
        self.path_column = path_column
        self.create_npy_files = create_npy_files
        self.extra_column_outputs = extra_column_outputs
        
        assert not batch_size % min_num_samples_per_class, "I'm afraid we're collating with a custom collation_fcn, and it would mean the *world* to us if you wouldn't create extra problems where it's avoidable..."
        self.max_num_classes_in_batch = batch_size // min_num_samples_per_class  
        
        self.epoch_len = epoch_len         
    
    def __len__(self):
        return self.epoch_len if self.epoch_len else len(self.meta_data)
    
    @staticmethod
    def filter_by_column(df, column_name, column_values=None):
        if column_values is None:
            return df
                
        if not (hasattr(column_values, '__contains__') and callable(column_values.__contains__)):
            column_values = [column_values]
            
        filtered_df = df[df[column_name].isin(column_values)]
        return filtered_df
            
    
    @staticmethod
    def create_class_ind_column(df, class_column):
        class_to_ind = {c: i for i, c in enumerate(sorted(set(df[class_column])))}
        df[class_column + IND_SUFFIX] = df[class_column].map(class_to_ind)
        return df
    
    
    @staticmethod
    def create_subset_column(df, class_column, subset):
        df[class_column + SUBSET_SUFFIX] = df[class_column].apply(lambda x: x if x in subset else "")
        return df
    
    
    def preprocess_class_columns(self):
        meta_data = self.meta_data
        meta_data = self.create_class_ind_column(meta_data, self.class_column)
        meta_data = self.create_subset_column(meta_data, self.class_column, self.class_subset)
        meta_data = self.create_class_ind_column(meta_data, self.class_column + SUBSET_SUFFIX)
        self.meta_data = meta_data   
        self.subset_class_column = self.class_column + SUBSET_SUFFIX + IND_SUFFIX
        self.class_column = self.class_column + IND_SUFFIX
    
    
    @staticmethod
    def get_or_create_npy_file(png_path):
        npy_path = png_path.replace('.png', '.npy')
        if osp.exists(npy_path):
            try:
                im = np.load(npy_path)
            except ValueError:
                im = imageio.imread(png_path)
            return im
        
        im = imageio.imread(png_path)
        np.save(npy_path, im)
        return im
    
    
    def crop_to_target_crop_size(self, x):
        if self.target_crop_size is not None:
            h, w = x.shape
            target_h, target_w = self.target_crop_size
            crop_h = h - target_h
            crop_w = w - target_w

            assert not crop_h % 2
            crop_h_ = crop_h // 2

            assert not crop_w % 2
            crop_w_ = crop_w // 2

            if crop_h_:
                x = x[crop_h_:-crop_h_, :]

            if crop_w_:
                x = x[:, crop_w_:-crop_w_]
                
            assert x.shape == self.target_crop_size
        return x
    
    def subsample_by_factor(self, x):
        if self.subsample_factor is not None:
            x = x[::self.subsample_factor[0], ::self.subsample_factor[1]]
        return x

    def row_to_sample(self, row):
        y = torch.tensor(row[self.class_column])
        y_subset = torch.tensor(row[self.subset_class_column])
        
        if self.create_npy_files:
            x = self.get_or_create_npy_file(row[self.path_column])
        else:
            x = imageio.imread(row[self.path_column])
            
        x = self.crop_to_target_crop_size(x)
        x = self.subsample_by_factor(x)
        
        x = torch.tensor(x)
        x = x.float().unsqueeze(0) / 255.
        x = self.transforms(x) * 255.

        extra_columns = {c: row[c] for c in self.extra_column_outputs}
        return x, y, y_subset, extra_columns, self.dataset_name
    
    
    def get_batch(self):
        if self.groupby_column is not None:
            groupby_val = random.choices(self.groupby_vals, self.groupby_weights)[0]
            meta_data = self.meta_data[self.meta_data[self.groupby_column] == groupby_val]
        else:
            meta_data = self.meta_data
            
        potential_classes = sorted(set(meta_data[self.class_column]))
        num_different_classes = len(potential_classes)
        num_classes_for_batch = min([self.max_num_classes_in_batch, num_different_classes])
        
        class_weights = np.array([self.class_weights_dict[c] for c in potential_classes])
        class_weights = class_weights / class_weights.sum()
        classes_for_batch = np.random.choice(potential_classes, size=num_classes_for_batch, replace=False, p=class_weights)
        # classes_for_batch = random.choices(potential_classes, class_weights, k=num_classes_for_batch)
        
        num_samples_per_class = {c: self.batch_size // num_classes_for_batch for c in classes_for_batch}
        rand_class_for_supplementing_batch_size = random.choice(classes_for_batch)
        num_samples_per_class[rand_class_for_supplementing_batch_size] = self.batch_size - sum(num_samples_per_class[c] 
                                                                                               for c in 
                                                                                               set(classes_for_batch) - {rand_class_for_supplementing_batch_size})
        
        classes_df = meta_data[meta_data[self.class_column].isin(classes_for_batch)]
        batch_df = classes_df.groupby(self.class_column).apply(lambda group: group.sample(n=num_samples_per_class[group.name], replace=True))  # TODO: better to solve without replace
        
        samples = []
        for _, row in batch_df.iterrows():
            samples.append(self.row_to_sample(row))
            
        return samples
    
            
    @staticmethod
    def collate_fn(data):
        data = data[0]
        x = torch.stack([d[0] for d in data])
        y = torch.stack([d[1] for d in data])
        y_subset = torch.stack([d[2] for d in data])
        extra_column_keys = data[0][3].keys()
        extra_columns = dict()
        for k in extra_column_keys:
            try:
                extra_columns[k] = torch.stack([torch.tensor(d[3][k]) for d in data]) 
            except TypeError:
                extra_columns[k] = [d[3][k] for d in data]
        dataset_name = tuple([data[0][4]] * len(data))
        
        batch = x, y, y_subset, extra_columns, dataset_name
        return batch
    
        
    def __getitem__(self, idx: int):
        if self.is_train:
            batch = self.get_batch()
            return batch
        else:            
            row = self.meta_data.iloc[idx]
            y = row[self.class_column]
            y_subset = row[self.subset_class_column]

            if self.create_npy_files:
                x = self.get_or_create_npy_file(row[self.path_column])
            else:
                x = imageio.imread(row[self.path_column])
            
            x = self.crop_to_target_crop_size(x)
            x = self.subsample_by_factor(x)
            
            x = torch.tensor(x)
            x = x.float().unsqueeze(0) / 255.
            x = self.transforms(x) * 255.
            sample = x, y, y_subset, {c: row[c] for c in self.extra_column_outputs}, self.dataset_name
            return sample

        
class PixelBioIDGeneralClassDatasetV2_TEMP(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 is_train: bool,
                 meta_data: pd.DataFrame,
                 class_column: str,
                 batch_size: int,
                 target_crop_size: Optional[tuple[int, int]] = None,
                 subsample_factor: Optional[tuple[int, int]] = None,
                 transforms: tuple[nn.Module, ...] = tuple(),
                 path_column: str = 'local_png_path',
                 create_npy_files: bool = False,
                 extra_column_outputs: list[str] = [],  # TODO: allow for string-valued columns
                 epoch_len: Optional[int] = 0,
                 ) -> None:
        super().__init__()
        
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.meta_data = meta_data
        if is_train:
            self.meta_data = self.filter_too_few_placements(self.meta_data)
        self.class_column = class_column
        self.batch_size = batch_size
        
        self.preprocess_class_columns()
        self.target_crop_size = target_crop_size
        self.subsample_factor = subsample_factor
        self.transforms = nn.Sequential(*transforms)
        self.path_column = path_column
        self.create_npy_files = create_npy_files
        self.extra_column_outputs = extra_column_outputs
                
        self.epoch_len = epoch_len
        
        if is_train:
            self.subj_sensor_pairs = sorted(set(map(tuple, 
                                                    self.meta_data[['subject_', 
                                                                    'sensor_serial']].drop_duplicates().values)))
            self.subj_sensor_to_sub_df = {key: group for key, group in 
                                          self.meta_data.groupby(['subject_', 'sensor_serial'])}
    
    def __len__(self):
        return self.epoch_len if self.epoch_len else len(self.meta_data)
    
    @staticmethod
    def filter_too_few_placements(df):
        how_many_placements = df.groupby(['subject_', 'sensor_serial'])['placement_ind'].nunique()
        keep_these_subj_sensor_pairs = how_many_placements[how_many_placements > 2].index.to_list()
        filt_df = df[df.apply(lambda r: (r['subject_'], r['sensor_serial']) in keep_these_subj_sensor_pairs, axis=1)]
        return filt_df        
    
    @staticmethod
    def filter_by_column(df, column_name, column_values=None):
        if column_values is None:
            return df
                
        if not (hasattr(column_values, '__contains__') and callable(column_values.__contains__)):
            column_values = [column_values]
            
        filtered_df = df[df[column_name].isin(column_values)]
        return filtered_df
            
    
    @staticmethod
    def create_class_ind_column(df, class_column):
        class_to_ind = {c: i for i, c in enumerate(sorted(set(df[class_column])))}
        df[class_column + IND_SUFFIX] = df[class_column].map(class_to_ind)
        return df    
    
    def preprocess_class_columns(self):
        meta_data = self.meta_data
        meta_data = self.create_class_ind_column(meta_data, self.class_column)
        self.meta_data = meta_data   
        self.class_column = self.class_column + IND_SUFFIX
    
    
    @staticmethod
    def get_or_create_npy_file(png_path):
        npy_path = png_path.replace('.png', '.npy')
        if osp.exists(npy_path):
            try:
                im = np.load(npy_path)
            except ValueError:
                im = imageio.imread(png_path)
            return im
        
        im = imageio.imread(png_path)
        np.save(npy_path, im)
        return im
    
    def crop_to_target_crop_size(self, x):
        if self.target_crop_size is not None:
            h, w = x.shape
            target_h, target_w = self.target_crop_size
            crop_h = h - target_h
            crop_w = w - target_w

            assert not crop_h % 2
            crop_h_ = crop_h // 2

            assert not crop_w % 2
            crop_w_ = crop_w // 2

            if crop_h_:
                x = x[crop_h_:-crop_h_, :]

            if crop_w_:
                x = x[:, crop_w_:-crop_w_]
                
            assert x.shape == self.target_crop_size
        return x
    
    def subsample_by_factor(self, x):
        if self.subsample_factor is not None:
            x = x[::self.subsample_factor[0], ::self.subsample_factor[1]]
        return x
    
    
    def row_to_sample(self, row):
        y = torch.tensor(row[self.class_column])
        y_subset = 1 * y + 0
        
        if self.create_npy_files:
            x = self.get_or_create_npy_file(row[self.path_column])
        else:
            x = imageio.imread(row[self.path_column])

        x = self.crop_to_target_crop_size(x)
        x = self.subsample_by_factor(x)
        
        x = torch.tensor(x)
        x = x.float().unsqueeze(0) / 255.
        x = self.transforms(x) * 255.

        extra_columns = {c: row[c] for c in self.extra_column_outputs}
        return x, y, y_subset, extra_columns, self.dataset_name
    
    
    def get_batch(self):        
        random_subj_sens_pairs = random.sample(self.subj_sensor_pairs, 22)
        cross_placements_df = pd.concat([self.subj_sensor_to_sub_df[x].groupby('placement_ind').sample(n=1)
                                         for x in random_subj_sens_pairs])
        batch_df = cross_placements_df.sample(n=self.batch_size)
        
        samples = [self.row_to_sample(row) for _, row in batch_df.iterrows()]
            
        return samples
    
            
    @staticmethod
    def collate_fn(data):
        data = data[0]
        x = torch.stack([d[0] for d in data])
        y = torch.stack([d[1] for d in data])
        y_subset = torch.stack([d[2] for d in data])
        extra_column_keys = data[0][3].keys()
        extra_columns = dict()
        for k in extra_column_keys:
            try:
                extra_columns[k] = torch.stack([torch.tensor(d[3][k]) for d in data]) 
            except TypeError:
                extra_columns[k] = [d[3][k] for d in data]
        dataset_name = tuple([data[0][4]] * len(data))
        
        batch = x, y, y_subset, extra_columns, dataset_name
        return batch
    
        
    def __getitem__(self, idx: int):
        if self.is_train:
            batch = self.get_batch()
            return batch
        else:            
            row = self.meta_data.iloc[idx]
            y = row[self.class_column]
            y_subset = 1 * y + 0

            if self.create_npy_files:
                x = self.get_or_create_npy_file(row[self.path_column])
            else:
                x = imageio.imread(row[self.path_column])
            
            x = self.crop_to_target_crop_size(x)
            x = self.subsample_by_factor(x)
                
            x = torch.tensor(x)
            x = x.float().unsqueeze(0) / 255.
            x = self.transforms(x) * 255.
            sample = x, y, y_subset, {c: row[c] for c in self.extra_column_outputs}, self.dataset_name
            return sample


class PixelBioIDGeneralClassDatasetMultiVsOtherForDemo(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 is_train: bool,
                 meta_data: pd.DataFrame,
                 class_column: str,
                 batch_size: int,
                 class_subset: Optional[list[str]] = None,
                 transforms: tuple[nn.Module, ...] = tuple(),
                 path_column: str = 'local_png_path',
                 create_npy_files: bool = False,
                 extra_column_outputs: list[str] = [], 
                 epoch_len: Optional[int] = 0,
                 ) -> None:
        super().__init__()
        
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.meta_data = meta_data
        self.class_column = class_column
        self.batch_size = batch_size
        self.class_subset = class_subset if class_subset is not None else sorted(set(meta_data[class_column]))
                
        self.preprocess_class_columns()
        
        subset_classes = sorted(set(self.meta_data[self.subset_class_column]))
        num_classes = len(subset_classes)
        num_samples_per_class = batch_size // num_classes
        num_for_other_class = batch_size - (num_classes - 1) * num_samples_per_class
        self.num_samples_per_class = {c: num_samples_per_class if c else num_for_other_class for c in subset_classes}

        self.transforms = nn.Sequential(*transforms)
        self.path_column = path_column
        self.create_npy_files = create_npy_files
        self.extra_column_outputs = extra_column_outputs
        
        self.epoch_len = epoch_len         
    
    def __len__(self):
        return self.epoch_len if self.epoch_len else len(self.meta_data)
    
    @staticmethod
    def filter_by_column(df, column_name, column_values=None):
        if column_values is None:
            return df
                
        if not (hasattr(column_values, '__contains__') and callable(column_values.__contains__)):
            column_values = [column_values]
            
        filtered_df = df[df[column_name].isin(column_values)]
        return filtered_df
            
    
    @staticmethod
    def create_class_ind_column(df, class_column):
        class_to_ind = {c: i for i, c in enumerate(sorted(set(df[class_column])))}
        df[class_column + IND_SUFFIX] = df[class_column].map(class_to_ind)
        return df
    
    
    @staticmethod
    def create_subset_column(df, class_column, subset):
        df[class_column + SUBSET_SUFFIX] = df[class_column].apply(lambda x: x if x in subset else "")
        return df
    
    
    def preprocess_class_columns(self):
        meta_data = self.meta_data
        meta_data = self.create_class_ind_column(meta_data, self.class_column)
        meta_data = self.create_subset_column(meta_data, self.class_column, self.class_subset)
        meta_data = self.create_class_ind_column(meta_data, self.class_column + SUBSET_SUFFIX)
        self.meta_data = meta_data   
        self.subset_class_column = self.class_column + SUBSET_SUFFIX + IND_SUFFIX
        self.class_column = self.class_column + IND_SUFFIX
    
    
    @staticmethod
    def get_or_create_npy_file(png_path):
        npy_path = png_path.replace('.png', '.npy')
        if osp.exists(npy_path):
            try:
                im = np.load(npy_path)
            except ValueError:
                im = imageio.imread(png_path)
            return im
        
        im = imageio.imread(png_path)
        np.save(npy_path, im)
        return im
    
    
    def row_to_sample(self, row):
        y = torch.tensor(row[self.class_column])
        y_subset = torch.tensor(row[self.subset_class_column])
        
        if self.create_npy_files:
            x = self.get_or_create_npy_file(row[self.path_column])
        else:
            x = imageio.imread(row[self.path_column])

        x = torch.tensor(x)
        x = x.float().unsqueeze(0) / 255.
        x = self.transforms(x) * 255.

        extra_columns = {c: row[c] for c in self.extra_column_outputs}
        return x, y, y_subset, extra_columns, self.dataset_name
    
    
    def get_batch(self):
        meta_data = self.meta_data        
        batch_df = meta_data.groupby(self.subset_class_column).apply(lambda group: group.sample(n=self.num_samples_per_class[group.name], replace=False))  # TODO: better to solve without replace
        assert len(batch_df) == self.batch_size
        
        samples = [self.row_to_sample(row) for _, row in batch_df.iterrows()]
        return samples
    
            
    @staticmethod
    def collate_fn(data):
        data = data[0]
        x = torch.stack([d[0] for d in data])
        y = torch.stack([d[1] for d in data])
        y_subset = torch.stack([d[2] for d in data])
        extra_column_keys = data[0][3].keys()
        extra_columns = dict()
        for k in extra_column_keys:
            try:
                extra_columns[k] = torch.stack([torch.tensor(d[3][k]) for d in data]) 
            except TypeError:
                extra_columns[k] = [d[3][k] for d in data]
        dataset_name = tuple([data[0][4]] * len(data))
        
        batch = x, y, y_subset, extra_columns, dataset_name
        return batch
    
        
    def __getitem__(self, idx: int):
        if self.is_train:
            batch = self.get_batch()
            return batch
        else:            
            row = self.meta_data.iloc[idx]
            y = row[self.class_column]
            y_subset = row[self.subset_class_column]

            if self.create_npy_files:
                x = self.get_or_create_npy_file(row[self.path_column])
            else:
                x = imageio.imread(row[self.path_column])
                
            x = torch.tensor(x)
            x = x.float().unsqueeze(0) / 255.
            x = self.transforms(x) * 255.
            sample = x, y, y_subset, {c: row[c] for c in self.extra_column_outputs}, self.dataset_name
            return sample

