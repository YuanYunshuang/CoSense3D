import uuid

import numpy as np

from cosense3d.carla.map_utils import *


class CarlaMapManager:
    def __init__(self, world, cfgs):
        self.world = world
        self.cfgs = cfgs
        self.carla_map = self.world.get_map()

    def generate_map_mata(self):
        # cross walks
        crosswalks = self.carla_map.get_crosswalks()
        crosswalks_dict = {}

        tmp_list = []
        for key_points in crosswalks:
            if (key_points.x, key_points.y, key_points.z) in tmp_list:
                crosswalk_id = uuid.uuid4().hex[:6].upper()
                cross_marking = np.array(tmp_list)
                bound = self.get_bounds(cross_marking, cross_marking)
                crosswalks_dict[crosswalk_id] = {'xyz': cross_marking, 'bound': bound}
                tmp_list = []
            else:
                tmp_list.append((key_points.x, key_points.y, key_points.z))

        # lanes
        lanes_dict = {}
        # list of all start waypoints in HD Map
        topology = [x[0] for x in self.carla_map.get_topology()]
        # sort by altitude
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        for (i, waypoint) in enumerate(topology):
            # unique id for each lane
            lane_id = uuid.uuid4().hex[:6].upper()
            intersection_flag = True if waypoint.is_intersection else False

            waypoints = [waypoint]
            nxt = waypoint.next(self.cfgs['lane_sample_resolution'])[0]
            # looping until next lane
            while nxt.road_id == waypoint.road_id \
                    and nxt.lane_id == waypoint.lane_id:
                waypoints.append(nxt)
                nxt = nxt.next(self.cfgs['lane_sample_resolution'])[0]

            # waypoint is the centerline, we need to calculate left and right lane mark
            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for
                            w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for
                             w in waypoints]
            # convert the list of carla.Location to np.array
            left_marking = list_loc2array(left_marking)
            right_marking = list_loc2array(right_marking)
            mid_lane = list_wpt2array(waypoints)
            bound = self.get_bounds(left_marking, right_marking)

            lanes_dict[lane_id] = {
                'is_intersection': intersection_flag,
                'left': left_marking,
                'middle': mid_lane,
                'right': right_marking,
                'bound': bound
            }

        self.crosswalks_dict = crosswalks_dict
        self.lanes_dict = lanes_dict
        self.global_bounds = self.get_global_bound()


    @staticmethod
    def get_bounds(left_lane, right_lane):
        """
        Get boundary information of a lane.

        Parameters
        ----------
        left_lane : np.array
            shape: (n, 3)
        right_lane : np.array
            shape: (n,3)
        Returns
        -------
        bound : np.array
        """
        x_min = min(np.min(left_lane[:, 0]),
                    np.min(right_lane[:, 0]))
        y_min = min(np.min(left_lane[:, 1]),
                    np.min(right_lane[:, 1]))
        z_min = min(np.min(left_lane[:, 2]),
                    np.min(right_lane[:, 2]))
        x_max = max(np.max(left_lane[:, 0]),
                    np.max(right_lane[:, 0]))
        y_max = max(np.max(left_lane[:, 1]),
                    np.max(right_lane[:, 1]))
        z_max = max(np.max(left_lane[:, 2]),
                    np.max(right_lane[:, 2]))

        bounds = np.asarray([[[x_min, y_min], [x_max, y_max], [z_min, z_max]]])

        return bounds

    def get_global_bound(self):
        bounds = np.concatenate([v['bound'] for k, v in self.crosswalks_dict.items()] +
                                [v['bound'] for k, v in self.lanes_dict.items()], axis=0)
        xy_min = np.min(bounds[:, 0, :], axis=0) - 20
        xy_max = np.max(bounds[:, 1, :], axis=0) + 20
        z_max = np.max(bounds[:, 2, 1])
        return xy_min.tolist() + [0.0] + xy_max.tolist() + [z_max]

