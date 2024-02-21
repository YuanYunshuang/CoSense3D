#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=MAX_SPLIT_SIZE_MB=256

torchrun \
--nproc_per_node=4 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/streamLTS_fcooper_dairv2x.yaml \
--mode train \
--gpus 2 \
--log-dir /koko/yunshuang/train_out \
--run-name StreamLTS_fcooper_dairv2x \
--batch-size 2 \
--n-workers 8 \
#--resume-from /koko/yunshuang/train_out/StreamLTS_seq8/epoch1.pth