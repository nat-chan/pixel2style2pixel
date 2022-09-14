# %%
#!/usr/bin/env python3
from pathlib import Path
import qrcode # 書き出し
from pyzbar import pyzbar
from PIL import Image
import numpy as np

# %%
if False:
    i = "/home/natsuki/mao_DBmiyuki/maoPR1.npz"
    s = "/home/natsuki/mao_DBmiyuki/maoPR1.png"
    bg = Image.open(s)
    fg = qrcode.make(i)
    fg = fg.resize((128, 128))
    bg.paste(fg, (0, 0))
    j = pyzbar.decode(bg)[0].data
    assert i == j
    bg.save("/home/natsuki/mao_DBmiyuki/maoQR1.png")
# %%

synth = list(Path("examples/synth").glob("*.png"))
for s in synth:
    i = int(s.stem.replace('seed', ''))
    bg = Image.open(s)
    fg = qrcode.make(i)
    fg = fg.resize((128, 128))
    bg.paste(fg, (0, 0))
    j = int(pyzbar.decode(bg)[0].data)
    assert i == j
    bg.save(str(s).replace("synth", "qr"))