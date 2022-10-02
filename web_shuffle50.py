#!/usr/bin/env python3
import sys
from typing import List
from pathlib import Path
import numpy as np
import argparse

from sklearn import metrics

def strip_seed(sname):
    return int(sname.split("/")[-1].split(".")[0][4:])

N = 50

synth = sorted(map(str, Path("examples/synth").glob("*.png")))[:N]
sim = sorted(map(str, Path("examples/sim").glob("*.png")))[:N]
ref = sorted(map(str, Path("examples/synth").glob("*.png")))[N:]
ver1 = [f"examples/sim_synth/{strip_seed(a)}_{strip_seed(b)}.png" for a, b in zip(sim, ref)]
s2p = [f"examples/pair50/{i}/a.png" for i in range(N)]
ver1_s2p = ver1 + s2p
np.random.RandomState(0).shuffle(ver1_s2p)
L, R = ver1_s2p[:N], ver1_s2p[N:]

title = sys.argv[0].lstrip("web_").rstrip(".py")
datasetlists: List[List[str]] = [L, R]
names: List[str] = list()
names = """
左
右
""".strip().split("\n")

jscode = """
function onclick_button(i, j){
    document.getElementById(`button_${i}_${j}`).style = 'background-color:cyan;';
    document.getElementById(`button_${i}_${1-j}`).style = '';
    document.getElementById(`select_${i}`).innerHTML = ['左', '右'][j];
    let not_selected = [];
    let code = "";
    for(let k=0; k<50; k++){
        let tmp_innerHTML = document.getElementById(`select_${k}`).innerHTML;
        if(tmp_innerHTML != '左' && tmp_innerHTML != '右'){
            not_selected.push(k);
        }else{
            code += +(tmp_innerHTML == '右');
        }
    }
    if(not_selected.length == 0){
        let encoded = "";
        for(let k=0; k<5; k++){
            encoded += parseInt(code.substring(k*10, (k+1)*10), 2);
            if(k != 5-1){
                encoded += ",";
            }
        }
        document.getElementById('input_top').value = encoded;
    }else{
        document.getElementById('input_top').value = '以下の番号がまだ未選択です'+not_selected.toString();
    }
    window.scrollBy(0,256);
    console.log(i, j, code);
}
""".strip()

th = '\n'.join(f"<th style='width:256px;height:50px;font-size:27px;'>{name}</th>" for name in names)
HTML = []
HTML.append(f"""
<!DOCTYPE html>
<html>
<head>
<title>{sys.argv[0].rstrip(".py")}</title>
<meta charset="utf-8"/>
<script type="text/javascript">
{jscode}
</script>
</head>
<body>
これからAIが生成した画像が2枚ずつ左と右に表示されます。<br>
貴方は2枚を見比べてより美しいと思った方を選択してください。<br>
この時、画像から何のAIが生成したか類推して選択するのはやめてください。<br>
あくまで貴方の主観でより美しいと思った方を選択してください。<br>
ファーストインプレッションが重要なので1枚につき10秒程度で選択ください。<br>
全部で50組を選択し終えたら、トップに戻るボタンを押して以下の<br>
テキストボックスに表示されるコードを私に教えることでアンケートは終了です<br>
<br>
<input id='input_top' style='width:512px;', value='まだ何も選択されていません'>
</input>
<table border=1 style='border: 1px solid black; border-collapse: collapse;'>
<tr>
<th>番号</th>
<th>選択</th>
{th}
</tr>
""")

for i, items in enumerate(zip(*datasetlists)):
    HTML.append("<tr>")
    HTML.append(f"<td>{i}</td>")
    HTML.append(f"<td id='select_{i}'>未選択</td>")
    for j, item in enumerate(items):
        HTML.append("".join([
            f"<td>",
                f"<button id='button_{i}_{j}' onclick='onclick_button({i}, {j})' style=''>",
                    f"<img src='{item}' loading='lazy' style='width:256px;height:256px;'>",
                f"</button>",
            f"</td>"]))
    HTML.append("</tr>")

HTML.append("""
</table>
<button id='button_bottom' onclick='window.scrollTo(0, 0);' style='width:512px;height:256px;'>
お疲れさまでした、トップに戻る
</button>
</body>
</html>
""")

def c2bl(c):
    b = int(c)
    bl = [(b>>i)&1 for i in range(10)]
    bl.reverse()
    return bl
def code2bl(code, d="|"):
    code = code.lstrip(d).rstrip(d)
    code = code.split(d)
    code = [c2bl(c) for c in code]
    code = sum(code, [])
    return code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default='')
    parser.add_argument('--d', type=str, default=',')
    parser.add_argument('--readlink', type=str, default="")
    parser.add_argument('--metrics', type=bool, default=False)
    args = parser.parse_args()
    if False: pass
    elif args.metrics:
        metrics = dict()
        with open("examples/shuffle50_L_clip.txt", "r") as f:
            L_clip = np.array([float(line.strip())for line in f.readlines()])
        with open("examples/shuffle50_R_clip.txt", "r") as f:
            R_clip = np.array([float(line.strip())for line in f.readlines()])
        with open("examples/shuffle50_codes.txt", "r") as f:
            codes = [str(line.strip())for line in f.readlines()]
        I = len(codes)
        X = np.array([code2bl(code, args.d) for code in codes])
        Y = (L_clip < R_clip).astype(int)
        EX = X.mean(axis=0) # E[bl]
        assert X.shape == (I, N)
        assert Y.shape == (N,)
        assert EX.shape == (N,)

        ok = 0
        for i in range(I): # x users
            for x, y in zip(X[i], Y): # x N
                if x == y: ok += 1
        metrics["metrics1"] = 100*ok/(I*N)
        metrics["metrics2"] = 100*((EX > .5) == Y).mean()

        """
        \sigma^2 = N^{-1} \sum_{i}( E[X]-X_i )^2
        """

        user_var = list()
        for Xi in X: # x users
            user_var.append(((EX-Xi)**2).mean())

        for k, v in metrics.items():
            print(f"{k}:", f"{v:.2f}")
        
        print("\sigma^2:", sum(user_var))
        print("user_var:")
        print("\n".join(f"{u:.2f}"[1:] for u in user_var))
    elif args.readlink != "":
        for path in globals()[args.readlink]: # L or R
            print(Path(path).absolute())
    elif args.code == '':
        print("\n".join(HTML))
    else:
        Xi = code2bl(args.code, args.d)
        assert len(Xi) == N
        for i, x in enumerate(Xi):
            print(i, ["左", "右"][x])
