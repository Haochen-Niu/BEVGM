# -*-coding:utf-8-*-
import cv2
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import os
import sys
from PIL import Image

sys.path.insert(0, os.getcwd())
from core.topo_match import build_affinity_mat
from models.my_test import appear_embed
from tools.utils import compute_score


def fv_gen_topo(fv, depth, seg, appearance, ipm_info, metadata, depth_thr, seg_vis=None, vis=False):
    G = nx.Graph()
    depth_valid = depth < depth_thr * 1.5

    if appearance is not None:
        model = appearance["model"]
        input_transform = appearance["input_transform"]
        device = appearance["device"]

    pan_seg = seg["panoptic_seg"]
    seg_info = seg["segments_info"]
    stuff = metadata["stuff"]
    stuff_colors = metadata["color"]
    objects = metadata["object"]
    background = metadata["background"]

    homo = ipm_info["homo"]
    mask = ipm_info["mask"][:, :, 0]
    wm, hm = ipm_info["outputSize"]
    fx, fy, cx, cy = ipm_info["camInfo"]
    w, h = ipm_info["outputRes"]
    pxPerM = ipm_info["pxPerM"]
    h_fv, w_fv = depth_valid.shape
    seg = pan_seg * depth_valid
    warpedImg = cv2.warpPerspective(seg, homo, (w, h), flags=cv2.INTER_NEAREST)
    warpedImg[mask] = 0
    bev = np.zeros((h, w, 3), dtype=np.uint8)  # vis
    bow = np.zeros(len(stuff))
    background_data = {key: None for key in background}
    node_id = 0
    Image_fv = Image.fromarray(fv)
    for seg_dict in seg_info:
        category_id = seg_dict["category_id"]
        id = seg_dict["id"]
        i_class = stuff[category_id]
        kernel = np.ones((10, 10), np.uint8)
        if i_class in objects:
            i_mask = seg == id
            d0 = None
            if i_class == "building":
                preproc_img = cv2.erode(i_mask.astype(np.uint8), kernel)
                preproc_img = cv2.dilate(preproc_img, kernel)
                v00, u00 = np.where(preproc_img == 1)
                v0 = v00[::10]
                u0 = u00[::10]
                # v10 = v00[::10]
                # u10 = u00[::10]
                d0 = depth[v0, u0] * (h + w) / 2 / 100
                appear_flag = True
            else:
                v00, u00 = np.where(i_mask)
                # v10, u10 = v00, u00
                v0, u0 = v00, u00
                appear_flag = False
            if v0.size == 0:
                continue
            if objects[i_class][0] == 1:
                labels = np.zeros_like(v0)
            else:
                if d0 is None:
                    dbscan = DBSCAN(eps=objects[i_class][1], min_samples=objects[i_class][2]).fit(
                        np.vstack((v0 * 0.2, u0 * 2)).T)
                else:
                    dbscan = DBSCAN(eps=objects[i_class][1], min_samples=objects[i_class][2]).fit(
                        np.vstack((v0, u0, d0)).T)
                labels = dbscan.labels_

            for i in range(labels.max() + 1):
                v = v0[labels == i]
                u = u0[labels == i]
                plt.scatter(u, v, s=1, c=np.random.rand(3).reshape(1, -1), label=i)
                v_node = np.mean(v)
                u_node = np.mean(u)
                if u_node > 0.999 * w_fv or u_node < 0.07 * w_fv:
                    continue
                z = np.mean(depth[v, u])
                if (z > hm * 0.9 or z < 0) and i_class == "building":
                    continue
                elif (z > hm * 0.75 or z < 0) and i_class != "building":
                    continue
                if appear_flag and appearance is not None:
                    min_row, min_col = np.min(v), np.min(u)
                    max_row, max_col = np.max(v), np.max(u)
                    fv_i = (fv * np.stack([preproc_img] * 3, axis=-1))[min_row:max_row, min_col:max_col, :]
                    appearance_node = appear_embed(model, fv_i, input_transform, device)
                else:
                    appearance_node = None
                x = (u_node - cx) * z / fx
                y = (v_node - cy) * z / fy
                if -wm / 2 < (x / 0.9) < wm / 2:
                    u_bev = int(w / 2 + x * pxPerM[0] + 0.5)
                    v_bev = int(h - z * pxPerM[0] + 0.5)
                    G.add_nodes_from([(node_id, {"class": i_class,
                                                 "center": (u_bev, v_bev),
                                                 "color": tuple(c / 255 for c in stuff_colors[category_id]),
                                                 # "area": seg_dict["area"],
                                                 "area": v.shape,
                                                 "appearance": appearance_node,
                                                 "fv_shape": [1, 1, fv.shape[0], fv.shape[1]],
                                                 "fv_center": (u_node, v_node),
                                                 "node_id": node_id
                                                 })])
                    node_id += 1
                    bow[category_id] += 1
        # plt.legend(loc=2)
        elif i_class in background:
            background_data[i_class] = warpedImg == id
            bev[warpedImg == id, :] = stuff_colors[category_id]

    # stage: generate edge
    background_num = {}
    conn = []
    for node1 in range(G.number_of_nodes() - 1):
        n1x, n1y = G.nodes[node1]["center"]  # (u_bev, v_bev)
        for node2 in range(node1 + 1, G.number_of_nodes()):
            n2x, n2y = G.nodes[node2]["center"]
            x, y = GenericBresenhamLine(round(n1x), round(n1y), round(n2x), round(n2y))
            background_num["length"] = len(x) + 1
            for background_name, bg_mask in background_data.items():
                if bg_mask is None:
                    background_num[background_name + "_num"] = 0
                else:
                    background_num[background_name + "_num"] = sum(bg_mask[y, x]) / background_num["length"]
            G.add_edge(node1, node2, **background_num)
            conn.append([node1, node2])
            conn.append([node2, node1])

    G.graph['conn'] = conn
    G.graph['bow'] = bow
    G.graph['fv'] = np.array(Image_fv)
    G.graph['bev'] = bev
    G.graph['seg'] = np.array(seg_vis) if seg_vis is not None else None

    methods = ["rrwm"]
    aff, node_aff, _, _, _, _ = build_affinity_mat(G, G, metadata, vis=vis)
    if aff is None:
        self_aff = None
        self_scores = {k: [0] for k in methods}
    else:
        match_result = np.eye(G.number_of_nodes())
        score = compute_score(match_result, aff, node_aff)
        self_aff = aff
        self_scores = {k: [score] for k in methods}
    G.graph["self_scores"] = self_scores
    G.graph["self_aff"] = self_aff

    return G


def GenericBresenhamLine(x1, y1, x2, y2, ):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    s1 = 1 if ((x2 - x1) > 0) else -1
    s2 = 1 if ((y2 - y1) > 0) else -1

    boolInterChange = False
    if dy > dx:
        dx, dy = dy, dx
        boolInterChange = True

    e = 2 * dy - dx
    x = x1
    y = y1
    points_x = []
    points_y = []

    for i in range(0, int(dx)):
        if e >= 0:
            if boolInterChange:
                x += s1
            else:
                y += s2
            e -= 2 * dx
        if boolInterChange:
            y += s2
        else:
            x += s1
        e += 2 * dy
        points_x.append(x)
        points_y.append(y)
    return points_x, points_y
