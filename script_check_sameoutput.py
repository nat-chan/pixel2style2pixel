# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('/home/natsuki/stylegan2-ada-pytorch')
from PIL import Image
import pickle
import numpy as np
import torch
import numpy as np
import inspect
fm = "/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl"
fw1 = "/data/natsuki/danbooru2020/shizuku/projected_w.npz"
fw2 = "/data/natsuki/danbooru2020/shizuku/maid_projected_w.npz"
fw3 = "/data/natsuki/danbooru2020/shizuku/muzu_projected_w.npz"
fa = "/data/natsuki/danbooru2020/a.pt"
# %%
with open(fm, 'rb') as f:
    G1 = pickle.load(f)['G_ema'].cuda()
# %%
from models.stylegan2.model import Generator
G2 = Generator(512, 512, 2).cuda()
G2.load_state_dict(torch.load(fa)["g_ema"])
# %%
with np.load(fw1) as data:
    w1 = torch.from_numpy(data["w"]).cuda()
with np.load(fw2) as data:
    w2 = torch.from_numpy(data["w"]).cuda()
# %%
synth_image = G1.synthesis(w1, noise_mode='const')
#synth_image, _ = G2(w1)
synth_image = (synth_image + 1) * (255/2)
synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
Image.fromarray(synth_image, 'RGB')
# %%