#!/bin/bash

PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt.yaml --mode train --resume-from /media/yuan/luna/streamLTS/LTS_dairv2x/epoch50.pth --batch-size 4
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_opv2vt.yaml --mode train --resume-from /media/yuan/luna/streamLTS/streamLTS_opv2v/epoch50.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fcooper_dairv2x.yaml --mode train --resume-from /media/yuan/luna/streamLTS/LTS_fcooper_dairv2x/epoch50.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fcooper_opv2vt.yaml --mode train --resume-from /media/yuan/luna/streamLTS/LTS_fcooper_opv2v/epoch50.pth


PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_attnfusion_opv2vt.yaml --mode train --resume-from /data/CoSense3D/cosense3d/logs/LTS_attnfusion_opv2v_v2/epoch49.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_attnfusion_dairv2xt.yaml --mode train --resume-from /data/CoSense3D/cosense3d/logs/LTS_attnfusion_dairv2x_v2/epoch49.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fpvrcnn_dairv2xt.yaml --mode train --resume-from /data/CoSense3D/cosense3d/logs/LTS_fpvrcnn_dairv2x/epoch20.pth