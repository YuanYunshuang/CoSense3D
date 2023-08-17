import glob, os, tqdm

from cosense3d.utils.misc import load_json, save_json
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs


class NuScenes:
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(self, cvt_label_dir, data_dir):
        assert os.path.dirname(cvt_label_dir) == os.path.dirname(data_dir)
        self.label_dir = os.path.basename(cvt_label_dir)
        self.data_dir = os.path.basename(data_dir)
        self.load_mata(cvt_label_dir)

    def load_mata(self, label_dir):
        meta_files = glob.glob(os.path.join(label_dir, "*.json"))
        self.meta = {}
        print("Loading meta files...")
        for f in tqdm.tqdm(meta_files):
            scene = os.path.basename(f)[:-5]
            self.meta[scene] = load_json(f)
        print("Done.")

    def to_cosense(self, out_dir):
        """Convert nuscenes data into cosense3d format"""
        for scene, cur_meta in self.meta.items():
            scene_dict = {}
            for info in cur_meta:
                frame = info['token']
                fdict = cs.fdict_template()
                cs.update_agent(fdict, agent_id=0, agent_type="cav")
                cs.remove_lidar_info(fdict, agent_id=0)
                for cam_id, img, intr, extr in zip(info['cam_ids'], info['images'],
                                                    info['intrinsics'], info['extrinsics']):
                    cs.add_cam_to_fdict(fdict, agent_id=0, cam_id=cam_id,
                                        filenames=[os.path.join(self.data_dir, img)],
                                        intrinsic=intr, extrinsic=extr)
                fdict['meta']['bev_maps'] = {
                    'bev': os.path.join(self.label_dir, scene, info['bev']),
                    'aux': os.path.join(self.label_dir, scene, info['aux']),
                    'visibility': os.path.join(self.label_dir, scene, info['visibility']),
                }
                scene_dict[frame] = fdict

            save_json(scene_dict, f"{scene}.json")


if __name__=="__main__":
    nuscene = NuScenes("/koko/nuScenes/cvt_labels_nuscenes_v2",
                       "/koko/nuScenes/keyframes")
    nuscene.to_cosense("/media/hdd/yuan/CoSense3D/dataset/metas/nuscenes")