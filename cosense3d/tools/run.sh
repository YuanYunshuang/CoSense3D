#!/bin/bash

PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt.yaml --mode train --run-name LTS-dair --batch-size 4
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt_roi_focal_loss.yaml --mode train --run-name LTS-dair-roi_focal --batch-size 4