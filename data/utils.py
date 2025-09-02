from .datasets import PixelBioIDGeneralClassDatasetV2_TEMP as PixelBioIDGeneralClassDatasetV2
from misc.utils import transforms_list_from_dict

def get_ds_via_cfg(cfg, df):
    transforms_list = transforms_list_from_dict(cfg.transforms)

    ds = PixelBioIDGeneralClassDatasetV2(dataset_name=cfg.name,
                                         is_train=cfg.is_train,
                                         meta_data=df,
                                         class_column=cfg.class_column,
                                         target_crop_size=cfg.get('target_crop_size'),
                                         subsample_factor=cfg.get('subsample_factor'),
                                         transforms=transforms_list,
                                         batch_size=cfg.batch_size,
                                         path_column=cfg.get('path_column', 'vast_png_path'),
                                         extra_column_outputs=cfg.get('extra_column_outputs', []),
                                         epoch_len=cfg.get('epoch_len'))

    return ds