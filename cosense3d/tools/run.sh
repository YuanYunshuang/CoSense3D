#!/bin/bash

PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt.yaml --mode train \
--resume-from /media/yuan/luna/streamLTS/LTS_dairv2x/epoch50.pth --batch-size 4
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt_roi_focal_loss.yaml --mode train \
--resume-from /media/yuan/luna/streamLTS/LTS_dairv2x/epoch50.pth --batch-size 4