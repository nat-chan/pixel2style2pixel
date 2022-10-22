# %%

from os import environ
environ["CPATH"] = f'/usr/local/cuda-10.2/targets/x86_64-linux/include:{environ["CPATH"]}'
environ["LD_LIBRARY_PATH"] = f'/usr/local/cuda-10.2/targets/x86_64-linux/lib:{environ["LD_LIBRARY_PATH"]}'
environ["PATH"] = f'/usr/local/cuda-10.2/bin:{environ["PATH"]}'
environ["CUDA_VISIBLE_DEVICES"]="5"
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import torch

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

N = 50
ref = sorted(map(str, Path("examples/synth").glob("*.png")))[N:]
white55_w = sorted(map(str, Path("examples/white55").glob("*.npz")))[:N]


# %%

from script_w2df import W2DF
w2df = W2DF()
G = w2df.net.decoder.G
# %%
outdir = Path("examples/white55_ref")
outdir.mkdir(exist_ok=True)
for i in range(N):
    w1 = torch.Tensor(np.load(white55_w[i])["w"]).cuda()
    w2 = G.map(strip_seed(ref[i]))
    q = 10
    w3 = torch.cat([w1[:, :q, :], w2[:, q:, :]], axis=1)
    img = G.synth(w3)
    img.save(outdir/f"{i}.png")