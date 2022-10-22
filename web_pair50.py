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

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

N = 50

synth = sorted(map(str, Path("examples/synth").glob("*.png")))[:N]
sim = sorted(map(str, Path("examples/sim").glob("*.png")))[:N]
ref = sorted(map(str, Path("examples/synth").glob("*.png")))[N:]
ver1 = [f"examples/sim_synth/{strip_seed(a)}_{strip_seed(b)}.png" for a, b in zip(sim, ref)]
s2p = [f"examples/pair50/{i}/a.png" for i in range(N)]
Tanpopo = [f"examples/petalica/t{i}.jpg" for i in range(N)]
Satsuki = [f"examples/petalica/s{i}.jpg" for i in range(N)]
Canna   = [f"examples/petalica/c{i}.jpg" for i in range(N)]
webtoon = [f"examples/webtoon/{i}_ai-painter.png" for i in range(N)]
clipstudio = [f"examples/clipstudio/{i}.png" for i in range(N)]


title = sys.argv[0].rstrip(".py")

models = ["ver1", "s2p", "webtoon", "clipstudio", "Tanpopo", "Satsuki", "Canna"]

datasetlists: List[List[str]] = [synth, sim, ref]+[globals()[m] for m in models]
names: List[str] = list()
names = """
線画抽出元の画像
入力する線画
塗り方のreference画像
Ours ver1 w/ref
style2paints v4.5 w/ref	
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
    parser.jupyter_argument("--metrics True")
    args = parser.parse_args()
    if False: pass
    elif args.metrics:
#        FIDs = calc_FID(models)
        print("| name |", " | ".join(models), "|")
        print("| - |", " | ".join(['-' for _ in models]), "|")
#        print("| FID↓ |", " | ".join([f"{np.array(i).mean():.2f}" for i in FIDs.values()]), "|")
        clip = list()
        for model in models:
            with open(f"examples/pair50_{model}_clip.txt", "r") as f:
                clip.append( np.array([float(line.strip())for line in f.readlines()]) )
        print("| CLIP winrate↑ | 100-x |", end="")
        for c in clip[1:]:
            winrate = 100*(clip[0] < c).mean()
            print(f" {winrate:.2f} |", end="")
        print()
        SSIM = [list() for _ in models]
        PSNR = [list() for _ in models]
        for paths in zip(synth, *[globals()[k]for k in models]):
            synth_arr = cv2.imread(paths[0], 0)
            for i, path in enumerate(paths[1:]):
                arr = cv2.imread(path, 0)
                SSIM[i].append( cv2.quality.QualitySSIM_compute(synth_arr, arr)[0][0] )
                PSNR[i].append( cv2.quality.QualityPSNR_compute(synth_arr, arr)[0][0] )
        
        print("| SSIM↑ |", " | ".join([f"{np.array(i).mean():.2f}" for i in SSIM]), "|")
        print("| PSNR↑ |", " | ".join([f"{np.array(i).mean():.2f}" for i in PSNR]), "|")

    elif args.readlink != "":
        for path in globals()[args.readlink]: # ver1 or s2p
            print(Path(path).absolute())
    else:
        print("\n".join(HTML))