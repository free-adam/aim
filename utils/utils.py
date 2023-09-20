import os, sys
import subprocess, shutil

def prepare_3d_warping(base_path):
    print('prepare 3d models ...')

    _3d_warping_path = f'{base_path}/_3d_warping'
    if not os.path.exists(_3d_warping_path):
        os.makedirs(_3d_warping_path) 
    
    pytorch3d_lite_path = f'{_3d_warping_path}/pytorch3d-lite'
    if not os.path.exists(pytorch3d_lite_path):
        subprocess.run(['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git', pytorch3d_lite_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    sys.path.append(pytorch3d_lite_path)

    midas_path = f'{_3d_warping_path}/MiDaS'
    if not os.path.exists(midas_path):
        subprocess.run(['git', 'clone', 'https://github.com/isl-org/MiDaS.git', midas_path, '-b', 'v3'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not os.path.exists(f'{midas_path}/dpt_large-midas-2f21e586.pt'):
        subprocess.run(['wget', 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt', '-P', midas_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
        subprocess.run(['mv', midas_path+'/utils.py', midas_path+'/midas_utils.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    sys.path.append(midas_path)

    dd_path = f'{_3d_warping_path}/disco-diffusion'
    if not os.path.exists(dd_path):
        subprocess.run(['git', 'clone', 'https://github.com/alembics/disco-diffusion.git', dd_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    sys.path.append(dd_path)
    
    adabins_path = f'{_3d_warping_path}/AdaBins'
    if not os.path.exists(adabins_path):
        subprocess.run(['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git', adabins_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not os.path.exists(f'{adabins_path}/AdaBins_nyu.pt'):
        subprocess.run(['gdown', '--id', '1GbWk_UBApgWjaW3Aqf6xJMI3OlysgI-E', '-O', f'{adabins_path}/AdaBins_nyu.pt'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    sys.path.append(adabins_path)


def load_prompts(prompts_file):
    print('load_prompts ...')
    # load text file.
    prompts_list = []
    with open(prompts_file) as f:
        for line in f:
            num, prompt = line.strip().split('#')       
            prompts_list.append(prompt.strip()) 
    return prompts_list
