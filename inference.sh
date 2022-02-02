#!/bin/bash

dir=/data/natsuki/danbooru2020/psp/encavgsim_1632393929

$1 scripts/inference.py \
    --exp_dir=${dir} \
    --checkpoint_path=${dir}/checkpoints/best_model.pt \
    --data_path=/data/natsuki/whitechest_sim_val \
    --test_batch_size=16 \
    --test_workers=8 \
    --latent_mask=2,3,4,5,6,7,8,9,10,11,12,13,14,15