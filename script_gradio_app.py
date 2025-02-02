# %%
#!/usr/bin/env python3
import gradio as gr
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
try:
    from script_imgtransform import imgtransform
except:
    imgtransform = lambda *x: x

def load_attr(i, device="cuda"):
    wm_np = np.load(f"/data/natsuki/fact_lib/i2v_wm/{i}.npy")
    wm_np = wm_np.astype(np.float32)
    wm = torch.from_numpy(wm_np[None,:][[0]*16,:]).to(device)
    return wm

# sketch_simplification
with working_dir("/home/natsuki/sketch_simplification"):
    import simplify
    sm = simplify.StateModel("model_gan")

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

def fn(content_image, style_image, attr_txt, attr_strength):
    if content_image is None: return None, "Sketch input is empty"
    if attr_txt == "": attr_txt = "なし"
    content_image, style_image = imgtransform(content_image, style_image)
    try:
        seed = int(pyzbar.decode(Image.fromarray(style_image))[0].data)
        if seed == 0:
            output_text = f"Success, now random seed mode"
        else:
            output_text = f"Successfully read seed={seed} from QR"
    except:
        seed = 0
        output_text = f"Failed to read QR, fall back to random seed"
    
    content_torch = torch.from_numpy(
        sm.normalize(content_image[None,None,:,:]/255)
    ).cuda().float()
    content_torch = sm.model(content_torch)
    content_torch = resize(content_torch)

    if seed == 0:
        latent_to_inject = None
    else:
        latent_to_inject = mapping(net.decoder.G, seed)

    replace_avg = ATTR.get(attr_txt, None)
    if replace_avg is not None:
        replace_avg = attr_strength*replace_avg + (1-attr_strength)*net.latent_avg

    output_torch = run_on_batch(content_torch, net, opts, latent_to_inject, replace_avg)
    output_pillow = common.tensor2im(output_torch[0])
    output_numpy = np.array(output_pillow)
    output_text += f" | attr={attr_txt}, strength={attr_strength}"
    now = datetime.now().strftime("%m/%d %H:%M %S")
    print(now, output_text)
    return output_numpy, output_text


ATTR = {
    "なし": None,
    "男の子(イケメン)": load_attr(24),
    "男の子(ショタ)": load_attr(83),
    "ケモミミ": load_attr(334),
    "ウサミミ": load_attr(133),
#    "メイド": load_attr(138),
    "眼鏡": load_attr(57),
    "目隠れ": load_attr(281),
    "ちびキャラ": load_attr(96),
    "ちび> <": load_attr(354),
    "ちびo o": load_attr(449),
    "ちび :3": load_attr(238),
    "満面の笑み(ハッピー)": load_attr(288),
    "ロングヘア": load_attr(2),
    "ショートヘア": load_attr(3),
    "ツインテ": load_attr(19),
    "ショートツインテ": load_attr(251),
    "三つ編み": load_attr(64),
#    "三つ編み(ツイン)": load_attr(114),
    "マフラー": load_attr(120),
    "メイド(服・髪飾り)": load_attr(207),
    "学生服": load_attr(21),
    "スーツ": load_attr(370),
    "フリル": load_attr(126),
    "エルフ": load_attr(482),
    "ロリータ服": load_attr(481),
    "洋画風": load_attr(191),
    "赤面": load_attr(4),
    "手を上にあげる": load_attr(234), 
    "年齢を下げる": load_attr(268),
}

content_image_input = gr.inputs.Image(label="スケッチの入力", shape=(512, 512), image_mode="L")
style_image_input = gr.inputs.Image(label="塗り方の入力※左上のQRコードから読み取ります", shape=(512, 512))
attribute_text_input = gr.inputs.Radio(label="属性の変更", choices=list(ATTR.keys()))
attribute_strength_input = gr.inputs.Slider(label="属性変更の強さ", minimum=0, maximum=1)
image_output = gr.outputs.Image(label="出力")
status_output = gr.outputs.Textbox(label="ステータス")

# Examples
qr = sorted(map(str, Path("examples/qr_jpg").glob("*.jpg"))) # relative path
sim = sorted(map(str, Path("examples/sim_jpg").glob("*.jpg")))[::-1]
_qr = sorted(map(str, Path("examples/_qr_jpg").glob("*.jpg")))
_sim = sorted(map(str, Path("examples/_sim_jpg").glob("*.jpg")))
random.shuffle(qr)
random.shuffle(sim)
examples = [list(e) for e in zip(_sim+sim, _qr+qr)]

iface = gr.Interface(
fn=fn,
inputs=[content_image_input, style_image_input, attribute_text_input, attribute_strength_input],
outputs=[image_output, status_output],
examples=examples,
examples_per_page=5,
allow_flagging="never",
title="スケッチからのキャラクタ生成",
description="必ず下までスクロールして説明を読んでからお使いください",
article="""
## **【初めに】**

使用している技術は[StyleGAN2](https://github.com/NVlabs/stylegan2)と[pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
の組み合わせで新規なものではないです。また入力例はStyleGAN2による生成画像と、そこから機械的に抽出されたスケッチです。
これらの画像と生成画像の営利目的や非常識な使用はおやめください。

## **【入力例から実行するやり方】**

- **入力例**から好きな<スケッチ, 塗り方>の組を選ぶ。
- **送信**ボタンを押す。
- 右上の**出力**に実行経過時間が表示されるのでしばらく待つ。
- **出力**コンポーネントに生成結果が表示される。

**入力例**の各列をクリックすると、その列の<スケッチ, 塗り方>の両方が入力コンポーネントに登録されます。
ペア以外の<スケッチ, 塗り方>の組で実行してみたい場合は以下の手順を踏んでください。

- 好きな「塗り方の入力」を**入力例**から選び、クリックする。
- 「スケッチの入力」コンポーネントにも画像がロードされるので**右上のXボタン**を押して消し、**ここに画像をドロップ-または-クリックしてアップロード**に切り替わる。
- 好きな「スケッチの入力」を**入力例**から選び、左クリックで新しいタブで開く。
- 新しいタブから「スケッチの入力」コンポーネントに選んだスケッチ画像をドラッグアンドドロップする。
    - 同じタブを2つ開いて片方は入力コンポーネントを、もう片方は**入力例**を表示しドラッグアンドドロップするのも速いです。

## **【自分で描いたスケッチで実行するやり方】**

- 自分で描いたスケッチで実行したい方は顔の位置や大きさを揃えてください。
- 平均画像をダウンロードしてペインティングソフトで透明度を薄くして
- 目の位置や顔の大きさをそろえるようにスケッチを描いて**正方形**で出力してください、推奨は512x512pxです。
<table>
  <tr>
    <th>RGB空間上の平均</th>
    <th>W空間上の平均</th>
  </tr>
  <tr>
    <td><img src=https://i.imgur.com/fkK07TJ.png></img></td>
    <td><img src=https://i.imgur.com/0XMoEtX.png></img></td>
  </tr>
</table>

- あとは「スケッチの入力」コンポーネントの右上のXボタンを押して消し**ここに画像をドロップ-または-クリックしてアップロード**に切り替える。
- 作成したスケッチをアップロードする。

""".strip()
)
iface.launch(share=True)

"""
TODO
/home/natsuki/miniconda3/envs/ada/lib/python3.8/site-packages/gradio/deprecation.py:40: UserWarning: `optional` parameter is deprecated, and it has no effect
warnings.warn(value)
/home/natsuki/miniconda3/envs/ada/lib/python3.8/site-packages/gradio/outputs.py:42: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components
warnings.warn(
/home/natsuki/miniconda3/envs/ada/lib/python3.8/site-packages/gradio/outputs.py:21: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components
warnings.warn(
/home/natsuki/miniconda3/envs/ada/lib/python3.8/site-packages/gradio/deprecation.py:40: UserWarning: The 'type' parameter has been deprecated. Use the Number component instead.
"""
