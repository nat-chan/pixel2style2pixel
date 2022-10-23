# %%
from os import environ
device = "cuda"
from nokogiri.working_dir import working_dir
from pathlib import Path
# sketch_simplification
with working_dir("/home/natsuki/sketch_simplification"):
    import simplify
    sm = simplify.StateModel("model_gan")
with working_dir("/home/natsuki/stylegan2-ada-pytorch"):
    from script_util import wrap_G
with working_dir("/home/natsuki/sketchKeras"):
    import pytorch_converted.model
    pmodel = pytorch_converted.model.Model().eval().requires_grad_(False).to(device)
# %%

from argparse import Namespace
import torch

from models.psp import pSp
from options.test_options import TestOptions
import kornia
from PIL import Image
import numpy as np

# %%
root = Path("/home/natsuki/encavgsim_1632393929")
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
wrap_G(net.decoder.G)


class W2DF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = net
        self.sim_norm = sm.normalize
        self.sim = sm.model
        self.pmodel = pmodel
        self.net.decoder.G.synth(self.net.decoder.G.map()) # Setting up Pytorch plugin
    def forward(self, w, imode="w", omode="df"):
        if imode == "w":
            from_mat = self.net.decoder.G.synth(w, retarr="torch")[:, :, [2,1,0]] # torch.uint8(512, 512, 3) 20.00~255.00
        elif imode == "illust(512,512,3)": #0~255
            from_mat = w
        else:
            from_mat = self.net.decoder.G.synth(self.net.decoder.G.map(), retarr="torch")[:, :, [2,1,0]] # torch.uint8(512, 512, 3) 20.00~255.00
        if imode not in ["sim(1,1,512,512)"]:
            from_mat = from_mat.permute(2, 0, 1) # torch.uint8(3, 512, 512) 20.00~255.00
            from_mat_BCHW = from_mat[:, None, :,:].float() #BxCxHxW
            blur = kornia.filters.gaussian_blur2d(from_mat_BCHW, (2*9+1, 2*9+1), (3, 3))[:,0,:,:] #torch.float32(3, 512, 512) 37.23~255.00
            highPass = (from_mat.int()-blur.int()).float()/128.0 # torch.float32(3, 512, 512) -1.00~1.10
            inp = highPass/highPass.amax(axis=(1,2))[:,None,None] # torch.float32(3, 512, 512) -0.91~1.00
            mat = self.pmodel(inp[:,:,:,None]) # torch.float32(3, 512, 512, 1) -0.20~1.26
            mat = mat.permute(3, 1, 2, 0)[0] # torch.float32(512, 512, 3) -0.21~1.27
            mat = mat.amax(2) # torch.float32(512, 512) -0.17~1.26
            mat[mat<0.18] = 0 # torch.float32(512, 512) 0.00~1.26
            mat = - mat + 1 # torch.float32(512, 512) -0.26~1.00
            mat = mat * 255.0 # torch.float32(512, 512) -65.57~255.00
            mat = torch.clamp(mat, 0, 255) # torch.float32(512, 512) 0.00~255.00
            sim_torch = self.sim_norm(mat[None,None,:,:]/255)  # torch.float32(1, 1, 512, 512) -11.26~0.39
            sim_torch = self.sim(sim_torch) # torch.float32(1, 1, 512, 512) 0.00~1.00
        if omode == "sim(1,1,512,512)":
            return sim_torch
        if imode == "sim(1,1,512,512)": #0~255
            sim_torch = w
        sim_bin = (sim_torch < 0.9).float()
        df = kornia.contrib.distance_transform(sim_bin)
        if omode == "df_sim":
            return df, sim_torch
        else:
            return df

if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    w2df = W2DF()
    w = net.decoder.G.map()

    with SummaryWriter(log_dir="./logs", comment="df") as writer:
        writer.add_graph(w2df, w, True)
    import IPython; IPython.embed()