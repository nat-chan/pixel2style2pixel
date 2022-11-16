from os import environ
environ["CPATH"] = f'/usr/local/cuda-10.2/targets/x86_64-linux/include:{environ["CPATH"]}'
environ["LD_LIBRARY_PATH"] = f'/usr/local/cuda-10.2/targets/x86_64-linux/lib:{environ["LD_LIBRARY_PATH"]}'
environ["PATH"] = f'/usr/local/cuda-10.2/bin:{environ["PATH"]}'
environ["CUDA_VISIBLE_DEVICES"]="6"
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from scripts.inference import run_on_batch
from utils import common
from tqdm import tqdm

datadir = Path("/data/natsuki")
outdir = Path("./examples")
N = 10**3

sim = sorted((datadir/"whitechest_sim_val").glob("**/*.png"))
assert 7499 == len(sim)
np.random.RandomState(0).shuffle(sim)
fold_sim = [sim[j*N:(j+1)*N] for j in range(len(sim)//N)]
random_seeds = np.random.RandomState(0).randint(2**32, size=N)
aesthetic_seeds = list()
with open("/home/natsuki/improved-aesthetic-predictor/cat_0_9.dat", "r") as f:
    for line in f:
        line = line.strip()
        score, seed = line.split()
        seed = int(seed)
        aesthetic_seeds.append(seed)
        if aesthetic_seeds == N:
            break

random_seeds = np.random.RandomState(0).randint(2**32, size=N)

from script_w2df import W2DF, opts
w2df = W2DF()
resize = transforms.Resize((256, 256))

def colorize(cname, seed):
    img = np.array(Image.open(cname))
    content_torch = torch.from_numpy(img[None, None, :, :]/255).cuda()
    content_torch = resize(content_torch)
    latent_to_inject = w2df.net.decoder.G.map(seed)
    output_torch = run_on_batch(content_torch, w2df.net, opts, latent_to_inject)
    output_pillow = common.tensor2im(output_torch[0])
    return output_pillow

outstem = ["fold0_v1_random", "fold0_v1_aesthetic"]
for ostem in outstem:
    (outdir/ostem).mkdir(exist_ok=True, parents=True)

for i in tqdm(range(N)):
    colorize(fold_sim[0][i], random_seeds[i]).save(outdir/outstem[0]/f"{i}.png")
    colorize(fold_sim[1][i], aesthetic_seeds[i]).save(outdir/outstem[1]/f"{i}.png")