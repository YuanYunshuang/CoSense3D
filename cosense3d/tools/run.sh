#!/bin/bash
set -e
#PYTHONPATH=/mars/projects20/CoSense3D

for ((i = 1; i <= 10; i += 1)); do
    echo Epoch $i
    python cosense3d/tools/test.py --log_dir /mars/projects20/logs/annealing-no_topk-no --ckpt epoch$i
done
