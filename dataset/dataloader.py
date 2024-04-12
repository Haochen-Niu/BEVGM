import os
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from tools.utils import gen_ipm_init_info


class BaseDataLoader:
    def __init__(self):
        self.dataset_path = None
        self.dataset_name = None
        self.transform_Twu_w = self.get_Twu_w()
        self.transform_Tc_cu = self.get_Tc_cu()
        self.ipm_info = None
        self.hm = None
        self.poses = None

    def get_dataset_info(self, dataset_path, dataset_name):
        raise NotImplementedError

    def get_ipm_init(self, r, wm, hm, path=None):
        self.ipm_info = gen_ipm_init_info(path, r, wm, hm)
        self.hm = hm

    def update_ipm(self, angle=None, euler_order="YXZ"):  # "YXZ"
        if angle is None:
            angle = [0, 0, 0]
        angle[0] = 0
        R_fv = Rotation.from_euler(euler_order, angle, degrees=True).as_matrix()

        K = self.ipm_info["K"]
        K_bev_inv = self.ipm_info["K_bev_inv"]
        normal = self.ipm_info["normal"]
        height = self.ipm_info["height"]
        Tw_fv = self.ipm_info["Tw_fv"]
        Tw_bev = self.ipm_info["Tw_bev"]

        Tw_fv[:3, :3] = R_fv
        Tfv_bev = np.linalg.inv(Tw_fv) @ Tw_bev

        R = Tfv_bev[0:3, 0:3]
        t = Tfv_bev[0:3, 3]
        homo_inv = K @ (R - t.reshape((3, 1)) @ normal.T / height) @ K_bev_inv
        homo = np.linalg.inv(homo_inv)

        self.ipm_info["homo"] = homo

    def get_timestamp_info(self):
        pass

    def get_Twu_w(self):
        """
        get transform_Twu_w  wu means original world coordination, u means predefined unity world coordination
         the predefined unity world coordination :
         same as the definition of cam coordination: x-right, y-down, z-forward
        :return:
        """
        return np.eye(4)

    def get_Tc_cu(self):
        """
        get transform_Tc_cu:c means original camera coordination, cu means standard camera coordination
        standard camera coordination: x - right, y - down, z - forward
        :return:
        """
        return np.eye(4)

    def load_rgb(self, timestamp):
        """
        load rgb image
        :param timestamp:
        :return:
        """
        raise NotImplementedError

    def load_depth(self, timestamp):
        """
        load depth image
        :param timestamp:
        :return:  pixel value means * meter
        """
        raise NotImplementedError

    def load_seg(self, timestamp):
        """
        load segmentation image
        :param timestamp:
        :return:
        """
        raise NotImplementedError

    def load_seg_vis(self, timestamp):
        """
        load segmentation image
        :param timestamp:
        :return:
        """
        raise NotImplementedError

    def get_cam_pose(self, timestamp, euler_order="YXZ"):
        pass

class SynthiaDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def get_dataset_info(self, dataset_path, dataset_name):
        self.dataset_name = dataset_name  # "7"
        self.dataset_path = os.path.join(dataset_path, "SYNTHIA-SEQS-02-" + dataset_name[:-1].upper())
        self.rgb_path_prefix = os.path.join(self.dataset_path, "RGB/Stereo_Left/Omni_" + self.dataset_name[-1])
        self.depth_path_prefix = os.path.join(self.dataset_path, "Depth/Stereo_Left/Omni_" + self.dataset_name[-1])
        self.seg_path_prefix = os.path.join(self.dataset_path,
                                            "RGB/Stereo_Left/Omni_" + self.dataset_name[-1] + "_seg_output")

    def get_ipm_init(self, r, wm, hm, path=None):
        if path is None:
            path = "config/cam_conf/SYNTHIA_cam.yaml"
        super().get_ipm_init(r, wm, hm, path)

    def get_timestamp_info(self):
        rgb_path = os.path.join(
            self.dataset_path, "RGB/Stereo_Left/Omni_" + self.dataset_name[-1])
        rgbs = os.listdir(rgb_path)
        rgbs.sort()
        timestamps = []
        poses = {}
        for name in rgbs:
            if name.split(".")[1] != "png":
                continue
            timestamp = name.split(".")[0]
            timestamps.append(timestamp)
            poses[timestamp] = np.loadtxt(
                os.path.join(self.dataset_path, "CameraParams", 'Stereo_Left/Omni_' + self.dataset_name[-1],
                             timestamp + ".txt")).reshape((4, 4)).T
        self.poses = poses
        return timestamps

    def get_Twu_w(self):
        transform_Tu_wu = np.eye(4)
        transform_Tu_wu[1, 1] = -1
        return transform_Tu_wu

    def get_Tc_cu(self):
        transform_Tc_cu = np.eye(4)
        transform_Tc_cu[1, 1] = -1
        transform_Tc_cu[2, 2] = -1
        return transform_Tc_cu

    def load_rgb(self, timestamp):
        rgb_path = os.path.join(self.rgb_path_prefix, timestamp + ".png")
        fv = np.array(Image.open(rgb_path).convert('RGB'))
        return fv

    def load_depth_gt(self, timestamp):
        depth_path = os.path.join(self.depth_path_prefix, timestamp + ".png")
        depth_ushort = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_ushort = np.array(depth_ushort)[:, :, 0]
        depth = depth_ushort / 100
        return depth

    def load_depth(self, timestamp):
        depth = self.load_depth_gt(timestamp)
        return depth

    def load_seg(self, timestamp):
        seg_path = os.path.join(self.seg_path_prefix, "data", timestamp + ".npy")
        seg = np.load(seg_path, allow_pickle=True).item()
        return seg

    def load_seg_gt(self, timestamp):
        mapping = {3: 0, 12: 0, 1: 10, 2: 2, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13, 9: 7, 10: 11, 15: 6}
        if timestamp == "000000":
            timestamp = "000001"
        seg_path = os.path.join(self.dataset_path, "GT/LABELS/Stereo_Left/Omni_" + self.dataset_name[-1],
                                timestamp + ".png")
        raw_label = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        label = raw_label[:, :, 2]
        panoptic_seg = np.ones_like(label) * 255
        segments_info = []
        i = 1
        for id_gt, id_city in mapping.items():
            pos = np.where(label == id_gt)
            area = len(pos[0])
            if area == 0:
                continue
            panoptic_seg[pos] = i
            if id_gt == 3:
                continue
            elif id_gt == 12:
                area = len(np.where(panoptic_seg == i)[0])
            segments_info.append({
                "id": i,
                "isthing": False,
                "category_id": id_city,
                "area": area
            })
            i += 1
        seg = {
            "panoptic_seg": panoptic_seg,
            "segments_info": segments_info
        }

        return seg

    def load_seg_vis(self, timestamp):
        seg_vis_path = os.path.join(self.seg_path_prefix, "vis", "panoptic_inference" + timestamp + ".png")
        seg_vis = Image.open(seg_vis_path)
        return seg_vis

    def get_cam_pose(self, timestamp, euler_order="YXZ"):
        Tw_c = self.poses[timestamp]
        Twu_cu = self.transform_Twu_w @ Tw_c @ self.transform_Tc_cu
        angle = Rotation.from_matrix(Twu_cu[:3, :3]).as_euler(euler_order, degrees=True)
        return Twu_cu, angle
