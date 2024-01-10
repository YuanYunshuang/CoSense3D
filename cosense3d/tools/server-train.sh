#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=MAX_SPLIT_SIZE_MB=256

torchrun \
--nproc_per_node=4 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/stream_lidar_st_v3.yaml \
--mode train \
--gpus 4 \
--data-path /koko/yunshuang/OPV2V/temporal \
--meta-path /koko/yunshuang/cosense3d/opv2v_temporal \
--log-dir /koko/yunshuang/train_out/cosense3d \
--run-name StreamLTSv3 \
--batch-size 2