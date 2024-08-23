import argparse, sys
import os

from cosense3d.config import load_yaml
from cosense3d.utils.misc import save_json
from cosense3d.utils.train_utils import seed_everything
from cosense3d.carla.map_manager import CarlaMapManager
from cosense3d.carla.scene_manager import get_scene_manager

sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla


class SimulationRunner:
    def __init__(self, args, cfgs):
        self.mode = args.mode
        self.cfgs = cfgs
        self.start_simulator()

    def start_simulator(self):
        # setup the carla client
        self.client = carla.Client('localhost', self.cfgs.get('client_port', 2000))
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if not self.world:
            sys.exit('World loading failed')

        # setup the new setting
        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = self.cfgs.get('sync_mode', True)
        new_settings.fixed_delta_seconds = self.cfgs.get('fixed_delta_seconds', 0.1)
        self.world.apply_settings(new_settings)

        # get managers
        self.spectator = self.world.get_spectator()

    def run(self):
        if self.mode == 'map':
            maps = os.listdir('../carla/assets/maps/png')
            bound_dict = {}
            for m in maps:
                town = m.split('.')[0]
                self.client.load_world(town)
                map_manager = CarlaMapManager(self.world, self.cfgs['map'])
                map_manager.generate_map_mata()
                bound_dict[town] = map_manager.global_bounds
                print(town, map_manager.global_bounds)
            save_json(bound_dict, '../carla/assets/map_bounds.json')
        elif self.mode == 'open_drive_map':
            maps = os.listdir('../carla/assets/maps/png')
            for m in maps:
                town = m.split('.')[0]
                self.client.load_world(town)
                open_drive_map = self.world.get_map().to_opendrive()
                with open(f'../carla/assets/maps/xodr/{town}.xodr', 'w') as fh:
                    fh.write(open_drive_map)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/carla.yaml")
    parser.add_argument("--mode", type=str, default="open_drive_map",
                        help="data | sim | map_meta | open_drive_map")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    seed_everything(2024)
    cfgs = load_yaml(args.config)
    sim_runner = SimulationRunner(args, cfgs)
    sim_runner.run()