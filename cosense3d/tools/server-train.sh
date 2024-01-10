#!/bin/bash

PYTHONPATH=.
OMP_NUM_THREADS=16

torchrun \
--nproc_per_node=2 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/stream_lidar_st_v2.yaml \
--mode train \
--gpus 2 \
--data_path /koko/yunshuang/OPV2V/temporal \
--meta_path /koko/yunshuang/cosense3d/opv2v_temporal