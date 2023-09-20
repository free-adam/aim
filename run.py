# read args.
import argparse
import configparser
import os   

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='xxx1', help='tag of this experiment')  
parser.add_argument('--param', type=str, default='param.ini', help='parameter file')
args = parser.parse_args()

# read parameters
base_path = os.path.dirname(__file__)
param_path = os.path.join(base_path, args.param)
param = configparser.ConfigParser()
param.read(param_path)

# make dirs
results_path = os.path.join(base_path, 'results', args.tag)
if not os.path.exists(results_path):
    os.makedirs(results_path)
  
# load prompts.
from utils import utils
prompts_list = utils.load_prompts(f"{base_path}/{param['sdxl']['prompts_file']}")

# prepare models
from utils import utils
utils.prepare_3d_warping(base_path)

# load packages
from diffusers import StableDiffusionXLPipeline
import torch
from _3d_warping import _3d_warping


midas_model, midas_transform = _3d_warping.init_midas_depth_model(f'{base_path}/_3d_warping/MiDaS/dpt_large-midas-2f21e586.pt', torch.device('cuda'))
camera_motion = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]

_3d_warping.do(base_path+'/results/xxx1/00000.jpg', camera_motion, midas_model, midas_transform, param, torch.device('cuda')).save(base_path+'/results/xxx1/00000_3d.jpg')

"""
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

for i in range(len(prompts_list)-19):
    image = pipe(prompt=prompts_list[i]).images[0]
    # file name begains from tag, four_numbers of i. jpg format
    image_filename = os.path.join(results_path, '{0:05d}.jpg'.format(i))
    image.save(image_filename)

print('done.')
"""