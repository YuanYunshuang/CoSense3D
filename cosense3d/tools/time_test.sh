#!/bin/bash


PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_opv2vt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_opv2v/epoch49.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fcooper_opv2vt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_fcooper_opv2v/epoch10.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_attnfusion_opv2vt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_attn_opv2v/epoch10.pth

PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_dairv2xt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_dairv2x/epoch47.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fcooper_dairv2xt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_fcooper_dairv2x/epoch10.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_attnfusion_dairv2xt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_attn_dairv2x/epoch10.pth


PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fpvrcnn_opv2vt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_fpvrcnn_opv2v/epoch10.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fpvrcnn_dairv2xt.yaml --mode train --resume-from /media/yuan/luna/LTS_time_test/LTS_fpvrcnn_dairv2x/epoch10.pth
