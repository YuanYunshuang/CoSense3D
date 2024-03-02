#!/bin/bash
set -e
PYTHONPATH=/mars/projects20/CoSense3D

#for ((i = 1; i <= 10; i += 1)); do
#    echo Epoch $i
#    python cosense3d/tools/test.py --log_dir /mars/projects20/logs/annealing-no_topk-no --ckpt epoch$i
#done

cd /mars/projects20/CoSense3D
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_opv2vt.yaml --mode train --resume-from /media/yuan/luna/streamLTS/LTS_dairv2x/epoch40.pth
