#!/usr/bin/env python3
from pathlib import Path
from pyzbar import pyzbar
from PIL import Image
from argparse import Namespace
from utils import common
import numpy as np
from options.test_options import TestOptions
from scripts.inference import run_on_batch
import torch
from models.psp import pSp
import torchvision.transforms as transforms
from nokogiri.working_dir import working_dir
from datetime import datetime
import random
from tqdm import tqdm

"""
styleganの生成画像100枚をsketch simplifyしたものを入力とし、
同じ画像群のwベクトルを塗りのreferenceとして100x100枚の画像をsim_synthに生成する
"""

#root = Path("/data/natsuki/danbooru2020/psp/encavgsim_1632393929")
root = Path("/home/natsuki/encavgsim_1632393929")
# encavg_1631706221 エンコード
epoch = "iteration_495000.pt"
ckpt = torch.load(root/f"checkpoints/{epoch}", map_location='cpu')
opts = ckpt['opts']
test_opts = TestOptions().parser.parse_args(
f"""
--exp_dir={root} \
--checkpoint_path={root}/checkpoints/{epoch} \
--data_path=/data/natsuki/whitechest_sim_val \
--test_batch_size=1 \
--test_workers=1 \
--latent_mask=10,11,12,13,14,15
""".split())
opts.update(vars(test_opts))
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False
if 'output_size' not in opts:
    opts['output_size'] = 1024
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()

def mapping(G, seed=1, psi=1):
    label = torch.zeros([1, G.c_dim], device="cuda")
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).cuda()
    w = G.mapping(z, label, truncation_psi=psi)
    return w

resize = transforms.Resize((256, 256))

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

def fn(cname, sname):
    seed = strip_seed(sname)
    img = np.array(Image.open(cname))
    content_torch = torch.from_numpy(img.transpose((2,0,1))[None,:1,:,:]/255).cuda()
    content_torch = resize(content_torch)
    latent_to_inject = mapping(net.decoder.G, seed)
    output_torch = run_on_batch(content_torch, net, opts, latent_to_inject)
    output_pillow = common.tensor2im(output_torch[0])
    return output_pillow

# Examples
sim = sorted(map(str, Path("examples/sim").glob("*.png")))
synth = sorted(map(str, Path("examples/synth").glob("*.png")))
assert len(sim) == 100
assert len(synth) == 100

output_dir = Path("./sim_synth")
output_dir.mkdir(exist_ok=True)
for cname in tqdm(sim):
    for sname in synth:
        cseed = strip_seed(cname)
        sseed = strip_seed(sname)
        output_pillow = fn(cname, sname)
        output_pillow.save(output_dir/f"{cseed}_{sseed}.png")