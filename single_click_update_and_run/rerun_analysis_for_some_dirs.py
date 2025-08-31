import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
cuda_visible_devices = [4]#list(range(8))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cuda_visible_devices)))
import sys
sys.path.append('/home/shlomi.fenster/notebooks/PixelBioID/V2/')
from pathlib import Path
from analyze_models_and_save_results import run_analysis_and_get_results


exps_dir = "/mnt/A3000/ML/Personalized/shlomi.fenster/PixelsBioID/V2/"

for datestr in ['18Jan2025']:#['17Dec2024', '31Dec2024']:
    exp_path = Path(exps_dir) / f'new_blueprint_data_16Dec2024_split__{datestr}'

    print(f'GONNA GET ME SOME ANALYSIS for {datestr}')
    full_test_df_file = '/mnt/A3000/ML/Personalized/shlomi.fenster/PixelsBioID/meta_data_dfs/split_16Dec2024_test_v0.pkl'
    analysis_results = run_analysis_and_get_results(exp_path, full_test_df_file, 4)
    print(f'ANALYSIS AQUIRED for {datestr}')

    analysis_results.to_pickle(exp_path / 'analysis_results.pkl')
    
print('ABSOLUTELY AND COMPLETELY DONE DUDE')

