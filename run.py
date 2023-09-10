# read args.
import argparse
import configparser
import os   

# load command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='xxx', help='tag of this experiment')  
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
print(prompts_list)

# prepare models
from utils import utils
utils.prepare_models(param)


# load packages
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

for i in range(len(prompts_list)):
    image = pipe(prompt=prompts_list[i]).images[0]
    # file name begains from tag, four_numbers of i. jpg format
    image_filename = os.path.join(results_path, '{0:05d}.jpg'.format(i))
    image.save(image_filename)

print('done.')
