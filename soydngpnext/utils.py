import os
from tqdm import tqdm
import requests
import torch
path = os.path.dirname(__file__)

# The utils module contains kinds of tools.

# In this module, the pointed file could be downloaded.
# more examples would be added in future
# the parameter path means the root path of downloaded files, for example, if you want to download to path 'onnx':
# --onnx
#   --model1.onnx
#   --model2.onnx
# you should assign it as 'onnx/'
def downloads(path, name):
    url = "http://xtlab.hzau.edu.cn/downloads/"
    # create the folder from parameter path, if this folder has already existed, do nothing
    os.makedirs(path, exist_ok=True)
    # the total size of this model
    response = requests.get(url + name)
    total_size = int(response.headers.get('content-length', 0))
    # download with progress bar by tqdm and requests
    with open(path + name, 'wb') as file, tqdm(
            desc=name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# This module is used for transforming .pt weights to .onnx weights for better performance.
# The inputPath should be like:
# ../
# inputPath/
#   -- weight1.pt
#   -- weightn.pt
# Results will be outputted to inputPath/../onnx

# transform .pt weights to .onnx weights, input_channel, input_W and input_H should be given
# set cuda switch True to load model and dummy tensor on cuda device, using cpu otherwise
def exportAsONNX(inputPath, input_W=206, input_H=206, input_channel=3, cuda=True):
    # path of the weights
    inputPath = inputPath
    # loop through files in the path
    for name in os.listdir(inputPath):
        # determine whether it is .pt weight
        if ".pt" in name:
            # print its name
            print(name)
            # load the weight
            model = torch.load(f"{inputPath}/{name}", map_location=torch.device('cuda' if cuda else 'cpu'))
            # generate a random input tensor
            dummy_input = torch.randn(1, input_channel, input_W, input_H).to('cuda' if cuda else 'cpu')
            # determine whether the output path exists
            if not os.path.exists(f"{inputPath}/../onnx"):
                # if not exists, create the folder
                os.mkdir(f"{inputPath}/../onnx")
            # export the model to .onnx, the first size is dynamic, which means it supports batch-input
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            torch.onnx.export(model, dummy_input, f"{inputPath}/../onnx/{name.replace('pt', 'onnx')}", input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)


# set a unified path name
# .
# --runs
#   --work
#     --work1
#     --workn
def outpath(work):
    # create runs/work/ path at first, if already exists, ignore this
    os.makedirs(f'{path}/runs/{work}/', exist_ok=True)
    # ergodic names of all folders
    folder_names = os.listdir(f'{path}/runs/{work}/')
    # look for the biggest index number
    if folder_names:
        number_list = [int(folder_name[len(work):]) for folder_name in folder_names if folder_name.startswith(work)]
        number = max(number_list) + 1
    else:
        number = 1
    new_path = f'{path}/runs/{work}/{work}{number}'
    os.makedirs(new_path)
    print(f"{work}ing results will be saved in {new_path}")
    return new_path
