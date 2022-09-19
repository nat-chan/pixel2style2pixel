#!/usr/bin/env python3
from subprocess import PIPE, Popen, STDOUT, check_output
from threading import Timer
from pathlib import Path
import os
import re
import coloredlogs, logging
try:
    from script_imgtransform import imgtransform
except:
    class imgtransform: pass
    imgtransform.notify = lambda txt: "failed!"
coloredlogs.install()
os.environ["PYTHONUNBUFFERED"] = "1"

cmd_app = ["/home/natsuki/miniconda3/envs/ada/bin/python3", "script_gradio_app.py"]
target_dir = Path("/home/natsuki/demo")
target_file = "index.html"

cmd_git = lambda prefix: f"""
cd {target_dir};
git add {target_file};
git commit -m "demo {prefix}";
git push;
""".strip()

html = lambda prefix: f"""
<!DOCTYPE html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta property="og:title" content="sketch2character" />
    <meta property="og:description" content="demo" />
    <meta property="og:image" content="https://i.imgur.com/UDO9PRn.png" />
    <title>sketch2character</title>
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-DENGG7LF7V"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-DENGG7LF7V');
    </script>
  </head>
  <body style="margin: 0px; padding: 0px; overflow: hidden">
    落ちた時は@mathbbQまで
    <iframe
      src="https://{prefix}.gradio.app"
      frameborder="0"
      style="
        overflow: hidden;
        overflow-x: hidden;
        overflow-y: hidden;
        height: 100%;
        width: 100%;
        position: absolute;
        top: 0px;
        left: 0px;
        right: 0px;
        bottom: 0px;
      "
      height="100%"
      width="100%"
    ></iframe>
  </body>
</html>
""".strip()

try:
    logging.info("app launch started")
    proc = Popen(
        cmd_app,
        stdout=PIPE,
        stderr=STDOUT,
        env=os.environ,
        close_fds=True, # BrokenPipeError
    )
    timer = Timer(12*60*60, proc.terminate)
    timer.start()
    for line in proc.stdout:
        line = line.decode().strip()
        print(line)
        if "Running on public URL" in line:
            prefix = int(re.findall(r"\d+", line)[0])
            logging.info(f"prefix={prefix} captured.")
            notify = imgtransform.notify(f"https://{prefix}.gradio.app")
            logging.info(f"notify={notify} send.")
            with open(target_dir/target_file, "w") as f:
                f.write(html(prefix))
            print(check_output(
                cmd_git(prefix),
                stderr=STDOUT,
                shell=True,
            ).decode())
        elif "Exception" in line:
            logging.warning(f"{line}")
            break
finally:
    proc.terminate()
    timer.cancel()