#!/usr/bin/env python3
import sys
from glob import glob
from typing import List
from pathlib import Path

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
print(f"""
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
	print("<tr>")
	print(f"<td>{i}</td>")
	for item in items:
		print(f"<td><img src='{item}' loading='lazy' style='width:256px;height:256px;'></td>")
	print("</tr>")

print("""
</table>
</body>
</html>
""")