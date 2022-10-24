#!/usr/bin/env python
# coding: utf-8

import os
device = "cuda"

import sys
import ast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nokogiri.working_dir import working_dir
import torch
import numpy as np
import torch
from pathlib import Path
import json
import random
import torchvision.transforms as transforms
from importlib import reload
from tqdm import tqdm
from functools import lru_cache
import time
from time import perf_counter
from datetime import timedelta



def make_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
make_deterministic()
resize = transforms.Resize((256, 256))

def mse(a, b):
    return float((a-b).square().mean())


w_achromatic = torch.tensor(np.load("/home/natsuki/stylegan2-ada-pytorch/achromatic_std0/whitechest_avg.npz")["w"], dtype=torch.float32, device=device)
fnames = sorted(Path("/home/natsuki/pixel2style2pixel/examples/sim").glob("*.png"))[:50]

@lru_cache(maxsize=None)
def g(fname):
    target_pillow = Image.open(fname)
    target_numpy = np.array(target_pillow) #f(512,512,3)
    target_torchCHW255 = torch.from_numpy(target_numpy).float().to(device).permute(2,0,1) #f(3,512,512)
    target_torchBCHW01 = target_torchCHW255[None,2:,:,:]/255 #Bのみ # f(1,1,512,512) 0~1
    target_df = w2df(target_torchBCHW01, imode="sim(1,1,512,512)")
    w_pSp = w2df.net.encoder(resize(target_torchBCHW01)) + w2df.net.latent_avg.unsqueeze(0)

    q = 10
    w_start = torch.cat([w_pSp[:,:q,:], w_achromatic[:,q:,:]], axis=1).detach()

    start_pillow = w2df.net.decoder.G.synth(w_start) # uint8(512, 512, 3) 33~255
    start_numpy =  np.array(start_pillow) #f(512,512,3)
    start_torchHWC255 = torch.from_numpy(start_numpy).float().to(device) #f(512,512,3)
    start_df = w2df(start_torchHWC255, imode="illust(512,512,3)")
    score_start = mse(target_df, start_df)
    return target_torchCHW255, w_start, score_start, start_numpy


def dirname(param):
    return "_".join(f"{k}={v}"for k, v in param.items())

def sub(
    j, 
    fname,
    root,
    dist_weight=.2,
    additional_weight=.8,
    w_std=0.2,
    custom_lr=0.1,
):

    stem = Path(fname).stem
    target_torchCHW255, w_start, score_start, start_numpy = g(fname)
    num_steps = 100
    projector = project(
        w2df,
        target=target_torchCHW255, # pylint: disable=not-callable
        device=device,
        num_steps=num_steps,
        verbose=False,
        dist_weight=dist_weight,
        additional_weight=additional_weight,
        w_avg = w_start,
        w_std=w_std,
        custom_lr=custom_lr,
    )

    w_mini = w_start
    score_mini = score_start
    synth_np_mini = start_numpy
    additional_dist_log = list()
    for data in tqdm(projector, total=num_steps, desc=f"{j}", dynamic_ncols=True):
        projected_w, synth_np, additional_dist = data 
        additional_dist_log.append(float(additional_dist))
        score_now = float(additional_dist)
        if score_now < score_mini:
            score_mini = score_now
            synth_np_mini = synth_np
            w_mini = projected_w
    Image.fromarray(synth_np_mini).save(root/f"{stem}.png")
    np.savez(root/f"{stem}.npz", w=w_mini.cpu().numpy())
    with open(root/f"{stem}.json", "w") as f:
        json.dump(additional_dist_log, f)
    with open(root/f"{stem}.txt", "w") as f:
        f.write(str(score_mini))
    return score_mini


def print_params():
    param = dict()
    for dist_weight in range(10):
        param["dist_weight"] = dist_weight/10
        param["additional_weight"] = (10-dist_weight)/10
        for custom_lr in [i/100 for i in range(1, 10+1)]:
            param["custom_lr"] = custom_lr
            for w_std in [i/100 for i in range(0, 100+1, 10)]+[float(i) for i in range(2, 10+1)]:
                param["w_std"] = w_std
                print(param)

def print_params2():
    param = dict()
    params = set()
    for dist_weight in range(0,10,2):
        param["dist_weight"] = dist_weight/10
        param["additional_weight"] = (10-dist_weight)/10
        for custom_lr in [i/100 for i in range(1, 10+1, 2)]:
            param["custom_lr"] = custom_lr
            for w_std in [i/100 for i in range(0, 100+1, 20)]:
                param["w_std"] = w_std
                params.add(repr(param))
    return params


def print_params3():
    param = dict()
    params = set()
    for dist_weight in range(0,10,1):
        param["dist_weight"] = dist_weight/10
        param["additional_weight"] = (10-dist_weight)/10
        for custom_lr in [i/100 for i in range(1, 10+1, 2)]: # lrはTOP8が0.05,0.03でバラつきが無いので探索幅を変えない
            param["custom_lr"] = custom_lr
            for w_std in [i/100 for i in range(0, 100+1, 10)]:
                param["w_std"] = w_std
                params.add(repr(param))
    return params
        
        

def main():
    params = [ast.literal_eval(line.strip()) for line in sys.stdin]
    for i, param in enumerate(params):
        start_time = time.perf_counter()
        print(f"\x1b[31;1m{i+1}/{len(params)}\x1b[m")
        root = Path(f"./grid100/{dirname(param)}")
        root.mkdir(exist_ok=True)
        scores = list()
        for j, fname in enumerate(fnames):
            scores.append( sub(j, fname, root, **param) )
        with open(root/"score.txt", "w") as f:
            f.write(str(sum(scores)/len(scores)))

        duration = timedelta(seconds=time.perf_counter()-start_time)
        print(f"\x1b[31;1mElapsed: {duration}\x1b[m")
        

if __name__ == "__main__":
    if 1 == 1:
        from script_w2df import W2DF
        w2df = W2DF()

        with working_dir("/home/natsuki/stylegan2-ada-pytorch"):
            import w2df_projector
            reload(w2df_projector)
            project = w2df_projector.project

    #for param in sorted(print_params3()-print_params2()):
    #    print(param)
    main()