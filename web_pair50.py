#!/usr/bin/env python3
import sys
from glob import glob
from typing import List
from pathlib import Path
from nokogiri import anyparse as argparse
import numpy as np
import cv2

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

N = 50

synth = sorted(map(str, Path("examples/synth").glob("*.png")))[:N]
sim = sorted(map(str, Path("examples/sim").glob("*.png")))[:N]
ref = sorted(map(str, Path("examples/synth").glob("*.png")))[N:]
ver1 = [f"examples/sim_synth/{strip_seed(a)}_{strip_seed(b)}.png" for a, b in zip(sim, ref)]
s2p = [f"examples/pair50/{i}/a.png" for i in range(N)]

title = sys.argv[0].rstrip(".py")
datasetlists: List[List[str]] = [synth, sim, ref, ver1, s2p]
names: List[str] = list()
names = """
線画抽出元の画像
入力する線画
塗り方のreference画像
Ours ver1 w/ref
style2paints v4.5 w/ref	
""".strip().split("\n")
"""
"""

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--readlink', type=str, default="")
    parser.add_argument('--metrics', type=bool, default=False)
    parser.jupyter_argument("--metrics True")
    args = parser.parse_args()
    if False: pass
    elif args.metrics:
        with open("examples/pair50_ver1_clip.txt", "r") as f:
            ver1_clip = np.array([float(line.strip())for line in f.readlines()])
        with open("examples/pair50_s2p_clip.txt", "r") as f:
            s2p_clip = np.array([float(line.strip())for line in f.readlines()])
        winrate = 100*(ver1_clip > s2p_clip).mean()
        print("| CLIP winrate↑ |", winrate, "|", 100-winrate, "|")
        SSIM = [list() for _ in range(2)]
        PSNR = [list() for _ in range(2)]
        for synth_path, ver1_path, s2p_path in zip(synth, ver1, s2p):
            synth_arr = cv2.imread(synth_path, 0)
            ver1_arr = cv2.imread(ver1_path, 0)
            s2p_arr = cv2.imread(s2p_path, 0)
            SSIM[0].append( cv2.quality.QualitySSIM_compute(synth_arr, ver1_arr)[0][0] )
            SSIM[1].append( cv2.quality.QualitySSIM_compute(synth_arr, s2p_arr)[0][0] )

            PSNR[0].append( cv2.quality.QualityPSNR_compute(synth_arr, ver1_arr)[0][0] )
            PSNR[1].append( cv2.quality.QualityPSNR_compute(synth_arr, s2p_arr)[0][0] )
        
        print("| SSIM↑ |", f"{np.array(SSIM[0]).mean():.2f}", "|", f"{np.array(SSIM[1]).mean():.2f}", "|")
        print("| PSNR↑ |", f"{np.array(PSNR[0]).mean():.2f}", "|", f"{np.array(PSNR[1]).mean():.2f}", "|")

    elif args.readlink != "":
        for path in globals()[args.readlink]: # ver1 or s2p
            print(Path(path).absolute())
    else:
        print("\n".join(HTML))