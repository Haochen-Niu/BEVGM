# -*-coding:utf-8-*-
import argparse
import math
import os
import pickle
import numpy as np
import yaml
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
from shapely.geometry import Polygon
import scipy.io as io


class Camera:
    K = np.zeros([3, 3])
    R = np.zeros([3, 3])
    t = np.zeros([3, 1])
    P = np.zeros([3, 4])

    def setK(self, fx, fy, cx, cy):
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy
        self.K[2, 2] = 1.0

    def setR(self, y, p, r):
        # Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
        # Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
        # Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
        # Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # switch axes (x = -y, y = -z, z = x)
        # self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))
        pass

    def setT(self, XCam, YCam, ZCam):
        X = np.array([XCam, YCam, ZCam])
        # self.t = -self.R.dot(X)
        self.t = X

    def updateP(self):
        Tw_c = np.zeros((4, 4))
        Tw_c[:3, :3] = self.R
        Tw_c[:3, 3] = self.t
        Tw_c[3, 3] = 1
        Rt = np.zeros([3, 4])
        Rt[0:3, 0:3] = self.R
        Rt[0:3, 3] = self.t
        self.T = Tw_c
        self.P = self.K.dot(Rt)

    def __init__(self, config):
        self.T = None
        self.setK(config["fx"], config["fy"], config["cx"], config["cy"])
        self.height = config["YCam"]
        if 'R' in config:
            self.R = np.resize(np.fromstring(config['R'], sep=' '), (3, 3))
        else:
            self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
        self.setT(config["XCam"], config["YCam"], config["ZCam"])
        self.updateP()


def gen_ipm_init_info(cam_config, r, wm, hm):
    """

    :param cam_config:
    :param r:
    :param wm:
    :param hm:
    :return:
    """
    with open(cam_config, 'r') as file:
        data = file.read()
        config = yaml.load(data, Loader=yaml.FullLoader)
    cam = Camera(config)

    if isinstance(r, list):
        pxPerM_w = r[0]
        pxPerM_h = r[1]
    else:
        pxPerM_w = r
        pxPerM_h = r
    W, H = int(wm * pxPerM_w), int(hm * pxPerM_h)
    Tw_bev = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, cam.height],
        [0, -1, 0, hm / 2],
        [0, 0, 0, 1]
    ])
    Tbev_w = np.linalg.inv(Tw_bev)
    K_bev = np.array([
        [pxPerM_w * (-cam.height), 0, W / 2],
        [0, pxPerM_h * (-cam.height), H / 2],
        [0, 0, 1]
    ])
    K_bev_inv = np.linalg.inv(K_bev)
    Tw_fv = cam.T
    normal = np.array([[0, 0, 1]]).T
    T_fv_bev = np.linalg.inv(Tw_fv) @ Tw_bev
    R = T_fv_bev[0:3, 0:3]
    t = T_fv_bev[0:3, 3]
    homo_inv = cam.K @ (R + t.reshape((3, 1)) @ normal.T / (-cam.height)) @ K_bev_inv
    homo = np.linalg.inv(homo_inv)
    mask = np.ones((H, W, 3), dtype=bool)
    for i in range(H):
        for j in range(W):
            theta = np.rad2deg(np.arctan2(j - W / 2, -i + H))  # (-Pi,Pi)
            if (config['fov'] / 2) > (theta - config["yaw"]) > - (config['fov'] / 2):
                mask[i, j, :] = False
    ipm_info = {
        "homo": homo,
        "Tw_fv": Tw_fv,
        "K_bev_inv": K_bev_inv,
        "Tw_bev": Tw_bev,
        "normal": normal,
        "K": cam.K,
        "height": cam.height,
        "mask": mask,
        "outputSize": [wm, hm],
        "camInfo": [config["fx"], config["fy"], config["cx"], config["cy"]],
        "outputRes": [W, H],
        "pxPerM": [pxPerM_w, pxPerM_h],
    }
    return ipm_info


def dict_from_txt(path, key_type=str):
    mapping = {}
    with open(path, "r") as f:
        for line in f.readlines():
            items = line.rstrip('\n').split(" ")
            assert len(items) >= 2
            key = key_type(items[0])
            val = list(map(float, items[1:])) if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping


def data_load(data_path, dataset, gt=False, half=False, rot=None, vis=False):
    if dataset["type"] == "kitti":
        data, vox_for_show = data_load_kitti_from_monoscene(data_path, dataset)
        return data, vox_for_show
    elif dataset["type"] == "airsim":
        filename = data_path
        seg_raw = Image.open(filename)
        full_bev = seg_raw.convert('RGBA')
        if rot is not None:
            seg_raw = seg_raw.rotate(rot)
        img = seg_raw
        scale_w = 0.8
        scale_h = 0.8
        if half:
            crop = (int((1 - scale_w) * (img.width / 2)), int((1 - scale_h) * (img.height / 2)),
                    int((1 + scale_w) * (img.width / 2)), int((img.height / 2)))
        else:
            crop = (int((1 - scale_w) * (img.width / 2)), int((1 - scale_h) * (img.height / 2)),
                    int((1 + scale_w) * (img.width / 2)), int((1 + scale_h) * (img.height / 2)))
        img = img.crop(crop)
        if half:
            img = img.resize((200, 100))
        else:
            img = img.resize((200, 200))
        img = np.array(img)
        resize = 2
        img_blender = Image.new('RGBA', (int(full_bev.width * resize), int(full_bev.height * resize)), (0, 0, 0, 150))
        img_blender = np.array(img_blender)
        img_blender[
        int(crop[1] + (resize - 1) * (full_bev.height / 2)):int(
            crop[3] + (resize - 1) * (full_bev.height / 2)),
        int(crop[0] + (resize - 1) * (full_bev.width / 2)):int(
            crop[2] + (resize - 1) * (full_bev.width / 2))
        ] = (0, 0, 0, 0)
        img_blender = Image.fromarray(img_blender)
        draw = ImageDraw.Draw(img_blender)
        p0 = (int(img_blender.width / 2), int(img_blender.height / 2))
        p1 = (img_blender.width, int(img_blender.height / 2 * (1 - math.sin(math.radians(30)))))
        p2 = (0, int(img_blender.height / 2 * (1 - math.sin(math.radians(30)))))
        draw.line([p2, p0, p1], fill=(255, 255, 255, 255), width=10)
        if rot is not None:
            img_blender = img_blender.rotate(-rot)
        crop = (int((resize - 1) * (full_bev.width / 2)), int((resize - 1) * (full_bev.height / 2)),
                int((resize + 1) * (full_bev.width / 2)), int((resize + 1) * (full_bev.height / 2)))
        img_blender = img_blender.crop(crop)
        full_bev = Image.alpha_composite(full_bev, img_blender).convert('RGB').resize((200, 200))
        data = np.zeros((len(dataset["class_names"]), img.shape[0], img.shape[1]))
        for i in range(len(dataset["class_names"])):
            mask = np.zeros(img.shape[:2])
            mask[np.where(np.all(img == tuple(dataset["color"][i]), axis=2))] = 1
            data[i, :, :] = mask  #
        img_for_show = [img, np.array(full_bev)]

        return data, img_for_show


def compute_score(X, affinity_matrix, node_aff):
    mask = node_aff >= 0
    X_mask = X * mask
    match_result = np.reshape(X_mask, (-1, 1), order='F')
    return float(np.matmul(np.matmul(match_result.T, affinity_matrix.numpy()), match_result))


def compute_PR(gt, pred):
    precision, recall, pr_thresholds = precision_recall_curve(gt, pred)
    auc_pr = average_precision_score(gt, pred)
    F1_score = 2 * precision * recall / (precision + recall + 10e-6)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    recall_p100 = recall[np.where(precision == 1)].max()
    precision_r100 = precision[np.where(recall == 1)].min()
    if recall_p100 > 0:
        threshold_p100 = pr_thresholds[np.where(precision == 1)[0][0]]
        right_ids_p100 = np.where(pred >= threshold_p100)
    else:
        right_ids_p100 = None
    return F1_max_score, precision, recall, auc_pr, pr_thresholds, precision_r100, recall_p100, right_ids_p100


def cal_comvis_area(t, angle, ranger_x=15, ranger_y=30, fov=100, vis=False):
    fov = math.radians(fov)
    angle = math.radians(angle)
    v1_1 = [0, 0]
    v1_2 = [ranger_x, ranger_x / math.tan(fov / 2)]
    v1_3 = [ranger_x, ranger_y]
    v1_4 = [-ranger_x, ranger_y]
    v1_5 = [-ranger_x, ranger_x / math.tan(fov / 2)]
    data1 = [v1_1, v1_2, v1_3, v1_4, v1_5]
    poly1 = Polygon(data1).convex_hull
    data2 = []
    for vertex in data1:
        data2.append([vertex[0] * math.cos(angle) - vertex[1] * math.sin(angle),
                      vertex[0] * math.sin(angle) + vertex[1] * math.cos(angle)] + t)
    poly2 = Polygon(data2).convex_hull
    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    if vis:
        fig, ax = plt.subplots()
        patch1 = plt_polygon(data1, color='g', alpha=0.5)
        patch2 = plt_polygon(data2, color='b', alpha=0.5)
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        plt.title("area_prop:{}".format(round(inter_area / poly1.area, 3)))
        ax.axis('auto')
        ax.set_aspect('equal')
        plt.show()
    return inter_area / poly1.area


def cal_comvis_area_oxford(t, angle, ranger_x=15, ranger_y=30, fov1=105.0, fov2=137.3, vis=False):
    # TODO: 2 polygons
    distance = math.sqrt(t[0] ** 2 + t[1] ** 2)
    comvis = math.exp(-abs(distance - 30))
    return comvis


def main():
    pass


if __name__ == '__main__':
    main()
