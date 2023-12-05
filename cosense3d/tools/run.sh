#!/bin/bash
set -e
PYTHONPATH=/mars/projects20/CoSense3D

#for ((i = 1; i <= 10; i += 1)); do
#    echo Epoch $i
#    python cosense3d/tools/test.py --log_dir /mars/projects20/logs/annealing-no_topk-no --ckpt epoch$i
#done

cd /mars/projects20/CoSense3D
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/voxelnet_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/voxelnet_all_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/voxelnet_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/voxelnet_ego_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/fpvrcnn_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/fpvrcnn_all_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/fpvrcnn_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/fpvrcnn_ego_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d_all_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d_ego_grad/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d_exp/epoch50.pth
#PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d_exp_no_downsample/epoch50.pth
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/sp3d_cav_v2.yaml --mode test --load-from /media/yuan/luna/cosense3d/sp3d_anchor/epoch50.pth
