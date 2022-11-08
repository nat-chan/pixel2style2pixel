#!/usr/bin/env python3
import sys
from glob import glob
from typing import List
from pathlib import Path
from nokogiri import anyparse as argparse
import numpy as np
import cv2
import subprocess
import pytorch_fid.fid_score
from pathlib import Path
from PIL import Image
import torch

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

N = 50

synth = sorted(map(str, Path("examples/synth").glob("*.png")))[:N]
sim = sorted(map(str, Path("examples/sim").glob("*.png")))[:N]
ref = sorted(map(str, Path("examples/synth").glob("*.png")))[N:]
ver1 = [f"examples/sim_synth/{strip_seed(a)}_{strip_seed(b)}.png" for a, b in zip(sim, ref)]
s2p = [f"examples/pair50/{i}/a.png" for i in range(N)]
webtoon = [f"examples/webtoon/{i}_ai-painter.png" for i in range(N)]
clipstudio = [f"examples/clipstudio/{i}.png" for i in range(N)]
Tanpopo = [f"examples/petalica/t{i}.jpg" for i in range(N)]
Satsuki = [f"examples/petalica/s{i}.jpg" for i in range(N)]
Canna   = [f"examples/petalica/c{i}.jpg" for i in range(N)]
white55 = sorted(map(str, Path("examples/white55").glob("*.png")))[:N]
grid100 = sorted(map(str, Path("grid100/dist_weight=0.3_additional_weight=0.7_custom_lr=0.05_w_std=0.7").glob("*.png")))[:N]
grid100ref = [f"grid100ref/dist_weight=0.3_additional_weight=0.7_custom_lr=0.05_w_std=0.7/{i}.png" for i in range(N)]
white55_ref = [f"examples/white55_ref/{i}.png" for i in range(N)]


title = sys.argv[0].rstrip(".py")

models = ["grid100", "white55", "grid100ref", "ver1", "s2p", "white55_ref", "webtoon", "clipstudio", "Tanpopo", "Satsuki", "Canna"]

datasetlists: List[List[str]] = [ref, synth, sim]+[globals()[m] for m in models]
names: List[str] = list()
names = """
塗り方のreference画像
線画抽出元の画像
入力する線画
pSp +achromatic +DF_proj (best) wo/ref
DF_proj from global_avg (.5,.5) wo/ref
pSp +achromatic +DF_proj (best) w/ref
pSp w/ref
style2paints v4.5 w/ref	
DF_proj from global_avg (.5,.5) w/ref
NAVER Webtoon AI Painter wo/ref
Celsys ClipStudioPaint v1.12.8 wo/ref
Pixiv(PFN) Petalica Paint Tanpopo wo/ref
Pixiv(PFN) Petalica Paint Satsuki wo/ref
Pixiv(PFN) Petalica Paint Canna wo/ref
""".strip().split("\n")

assert len(datasetlists) == len(names)

th = '\n'.join(f"<th style='width:256px;height:50px;font-size:27px;'>{name}</th>" for name in names)
HTML = list()
HTML.append(f"""
<!DOCTYPE html>
<html>
<head>
<title>{sys.argv[0].rstrip(".py")}</title>
<meta charset="utf-8"/>
</head>
<body>
<table border=1 style='border: 1px solid black; border-collapse: collapse;'>
<tr>
<th>id</th>
{th}
</tr>
""")

for i, items in enumerate(zip(*datasetlists)):
    HTML.append("<tr>")
    HTML.append(f"<td>{i}</td>")
    for item in items:
        HTML.append(f"<td><img src='{item}' loading='lazy' style='width:256px;height:256px;'></td>")
    HTML.append("</tr>")

HTML.append("""
</table>
</body>
</html>
""")

def calc_FID(models):
    # FID
    items = ["ref", *models]
    retval = dict()
    for item in items:
        Path(f"examples/{item}").mkdir(exist_ok=True)
        for i, src in enumerate(globals()[item]):
            src = Path(src).absolute()
            dst = Path(f"examples/{item}/{i}.png").absolute()
            if src != dst:
                print(subprocess.check_output(f"ln -sf {src} {dst}", shell=True).decode(), end="")
        if item == "ref": continue 
        retval[item] = float(subprocess.check_output(
                f"python -m pytorch_fid ./examples/ref ./examples/{item}",
                shell=True
            ).decode().lstrip("FID:").strip())
    return retval

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--readlink', type=str, default="")
    parser.add_argument('--metrics', type=bool, default=False)
    parser.add_argument('--DFMSE', type=bool, default=False)
    parser.add_argument('--FID', type=bool, default=False)
    parser.jupyter_argument("--metrics True")
    args = parser.parse_args()
    if False: pass
    elif args.metrics:
        if args.FID:
            FIDs = calc_FID(models)
        if args.DFMSE:
            from script_w2df import W2DF
            w2df = W2DF()
        print("| name |", " | ".join(models), "|")
        print("| - |", " | ".join(['-' for _ in models]), "|")
        if args.FID:
            print("| FID↓ |", " | ".join([f"{np.array(i).mean():.2f}" for i in FIDs.values()]), "|")
        clip = list()
        for model in models:
            clip_txt = Path(f"examples/pair50_{model}_clip.txt")
            if clip_txt.exists():
                with open(clip_txt, "r") as f:
                    clip.append( np.array([float(line.strip())for line in f.readlines()]) )
            else:
                clip.append(None)
        print("| CLIP winrate↑ |", end="")
        for i, c in enumerate(clip):
            q = 3
            if i == q:
                if c is not None:
                    winrate = 100*(clip[q] < c).mean()
                    print(f" {winrate:.2f} |", end="")
                else:
                    print(f" 100-x |", end="")
        print()
        SSIM = [list() for _ in models]
        PSNR = [list() for _ in models]
        DFMSE = [list() for _ in models]
        for paths in zip(sim, synth, *[globals()[k]for k in models]):
            if args.DFMSE:
                # TODO simを使わずに完成イラストから線画抽出，線画単純化をしてる．ユーザ等の完成イラストがない場合の比較には工事が必要．
#                sim_torch = torch.Tensor(np.array(Image.open(paths[0]).convert("L")))[None,None,:,:].cuda()
                synth_torch = torch.Tensor(np.array(Image.open(paths[1]))).cuda() # torch.uint8(512, 512, 3)
            synth_arr = cv2.imread(paths[1], 0) # uint8(512, 512)
            for i, path in enumerate(paths[2:]):
                arr = cv2.imread(path, 0)
                SSIM[i].append( cv2.quality.QualitySSIM_compute(synth_arr, arr)[0][0] )
                PSNR[i].append( cv2.quality.QualityPSNR_compute(synth_arr, arr)[0][0] )
                if args.DFMSE:
                    illust_torch = torch.Tensor(np.array(Image.open(path))).cuda() # torch.uint8(512, 512, 3)
                    sim_df = w2df(synth_torch, imode="illust(512,512,3)")[0,0,:,:]
                    illut_df = w2df(illust_torch, imode="illust(512,512,3)")[0,0,:,:]
                    DFMSE[i].append( (sim_df-illut_df).square().mean().cpu().detach().numpy() )
                    if i == 0:
                        Image.fromarray(sim_df.cpu().detach().numpy().astype(np.uint8)).save("sim_df.jpg")
                        Image.fromarray(illut_df.cpu().detach().numpy().astype(np.uint8)).save("illut_df.jpg")
        
        if args.DFMSE:
            print(
            "| DF-MSE↓ |", " | ".join([f"{np.array(line).mean():.2f}" for line in DFMSE]), "|")
        print("| SSIM↑ |", " | ".join([f"{np.array(line).mean():.2f}" for line in SSIM]), "|")
        print("| PSNR↑ |", " | ".join([f"{np.array(line).mean():.2f}" for line in PSNR]), "|")

    elif args.readlink != "":
        for path in globals()[args.readlink]: # ver1 or s2p
            print(Path(path).absolute())
    else:
        print("\n".join(HTML))