#!/bin/bash

# Training the pSp Encoder
mode=encavg

$1 scripts/train.py \
    --exp_dir=/data/natsuki/danbooru2020/psp/${mode}_$(date +%s) \
    --batch_size=16 \
    --test_batch_size=16 \
    --workers=8 \
    --test_workers=8 \
    --val_interval=2500 \
    --save_interval=5000 \
    --encoder_type=GradualStyleEncoder \
    --lpips_lambda=0.8 \
    --l2_lambda=1 \
    --id_lambda=0 \
    --moco_lambda=0.5 \
    --output_size=512 \
    --start_from_latent_avg \
    --stylegan_weights=/data/natsuki/training116/00023-white_yc05_yw04-mirror-auto4-gamma10-noaug/network-snapshot-021800.pkl \
    --dataset_type=whitechest \
    --use_wandb
#    --dataset_type=debug
#    --stylegan_weights=/data/natsuki/danbooru2020/a.pt
#    --start_from_latent_avg \
#    --input_nc=1