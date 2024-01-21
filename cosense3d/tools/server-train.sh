#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=MAX_SPLIT_SIZE_MB=256

torchrun \
--nproc_per_node=4 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/stream_lidar_st_v6.yaml \
--mode train \
--gpus 4 \
--data-path /koko/yunshuang/OPV2V/temporal \
--meta-path /koko/yunshuang/cosense3d/opv2v_temporal \
--log-dir /koko/yunshuang/train_out \
--run-name StreamLTS_seq8 \
--batch-size 1 \
--n-workers 8 \
--resume-from /koko/yunshuang/train_out/StreamLTS_seq8