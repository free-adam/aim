def gitclone(url, target_dir=None, branch_arg=None):
    run_args = ['git', 'clone']
    if branch_arg:
        run_args.extend(['-b', branch_arg])
    run_args.append(url)
    if target_dir:
        run_args.append(target_dir)
    res = subprocess.run(run_args, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipi(modulestr):
    res = subprocess.run(['pip', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipie(modulestr):
    res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def prepare_models(param):
    print('prepare_models ...')
    return

def load_prompts(prompts_file):
    print('load_prompts ...')
    # load text file.
    prompts_list = []
    with open(prompts_file) as f:
        for line in f:
            num, prompt = line.strip().split(':')       
            prompts_list.append(prompt.strip()) 
    return prompts_list
