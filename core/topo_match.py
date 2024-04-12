# -*-coding:utf-8-*-
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygmtools as pygm
import torch
from LinSATNet import linsat_layer

from tools.utils import compute_score

pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)


def node_aff_fn(node1, node2):
    if node1["class"] == node2["class"]:
        if node1["class"] == "building" and node1["appearance"] is not None:
            a = 0.2
            sim_class = math.exp(-a * (np.linalg.norm(node1["appearance"] - node2["appearance"])))
        else:
            sim_class = 0.005
    else:
        sim_class = 0
    return sim_class


def edge_aff_fn(edge1, edge2):
    sim = 10e-6
    for k in edge1.keys():
        if k == "length":
            if edge1[k] == 0:
                sim_len = 1
                continue
            sim_len = math.exp(- abs(edge1[k] - edge2[k]) / edge1[k])
        elif k == "road_num":
            if edge1[k] > 0.1 and edge2[k] > 0.1:
                sim_road = 1
            else:
                sim_road = 0
            sim += (edge1[k] - edge2[k]) ** 2
        else:
            sim += (edge1[k] - edge2[k]) ** 2
    sim /= (len(edge1.keys()) - 1)
    return (sim_len - sim) * 0.1 * sim_road


def my_aff_mat_from_node_edge_aff(node_aff, edge_aff, conn1, conn2, n1, n2, ne1, ne2, BACKEND):
    n1 = torch.tensor(n1).unsqueeze(0)
    n2 = torch.tensor(n2).unsqueeze(0)
    conn1 = torch.tensor(conn1).unsqueeze(0)
    conn2 = torch.tensor(conn2).unsqueeze(0)
    ne1 = torch.tensor(ne1).unsqueeze(0)
    ne2 = torch.tensor(ne2).unsqueeze(0)
    node_aff = node_aff.unsqueeze(0)
    if ne1 * ne2 == 0:
        edge_aff = None
    else:
        edge_aff = edge_aff.unsqueeze(0)

    aff = pygm.utils._aff_mat_from_node_edge_aff(node_aff, edge_aff, conn1, conn2, n1, n2, ne1, ne2, BACKEND)

    return aff.squeeze(0)


def build_affinity_mat(G1, G2, dataset, cons_flag=False, vis=False, match=None):
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    n1_center = np.zeros((n1, 2))
    n2_center = np.zeros((n2, 2))
    conn1 = G1.graph["conn"]
    conn2 = G2.graph["conn"]
    ne1 = len(conn1)
    ne2 = len(conn2)
    node_aff = torch.zeros(n1, n2)
    edge_aff = torch.zeros(ne1, ne2)

    if n1 * n2 == 0:
        return None, None, None, None, None, None

    for i in range(n1):
        G1_node = G1.nodes[i]
        n1_center[i, :] = G1_node["center"]
        for j in range(n2):
            G2_node = G2.nodes[j]
            n2_center[j, :] = G2_node["center"]
            node_aff[i, j] = node_aff_fn(G1_node, G2_node)
    node_aff_np = node_aff.numpy()

    if ne1 * ne2 == 0:
        edge_aff = None
        edge_aff_np = None
    else:
        for i in range(0, ne1, 2):
            for j in range(0, ne2, 2):
                edge_aff[i:i + 2, j:j + 2] = edge_aff_fn(G1.edges[conn1[i]], G2.edges[conn2[j]])
        edge_aff_np = edge_aff.numpy()

    aff = my_aff_mat_from_node_edge_aff(node_aff, edge_aff, conn1, conn2, n1, n2, ne1, ne2, pygm.BACKEND)
    if vis:
        plt.figure(figsize=(12, 4))
        if node_aff is not None:
            plt.subplot(1, 3, 1)
            plt.title(f'Node Affinity Matrix (size: {node_aff.shape[0]}$\\times${node_aff.shape[1]})')
            plt.imshow(node_aff_np, cmap='Blues')

        if edge_aff is not None:
            plt.subplot(1, 3, 2)
            plt.title(f'Edge Affinity Matrix (size: {edge_aff.shape[0]}$\\times${edge_aff.shape[1]})')
            plt.imshow(edge_aff_np, cmap='Blues')

        if aff is not None:
            plt.subplot(1, 3, 3)
            plt.title(f'Affinity Matrix (size: {aff.shape[0]}$\\times${aff.shape[1]})')
            plt.imshow(aff.numpy(), cmap='Blues')
    if ~torch.any(aff):
        aff = None
    if cons_flag:
        A = torch.zeros(n1 + n2, n1 * n2, dtype=torch.float32)
        b = torch.zeros(n1 + n2, dtype=torch.float32)

        for cons_id in range(n1 + n2):
            tmp = torch.zeros(n1, n2, dtype=torch.float32)
            if cons_id < n1:
                tmp[cons_id, 0:n2] = 1
            else:
                tmp[0:n1, cons_id - n1] = 1
            A[cons_id, :] = tmp.reshape(-1)
            b[cons_id] = 1

        E = torch.zeros(1, n1 * n2, dtype=torch.float32)
        f = None
        tmp = node_aff.reshape(-1)

        for i in range(n1 * n2):
            if tmp[i] <= 0:
                E[0, i] = 1
                f = torch.zeros(1, dtype=torch.float32)

        cons = {
            "A": A,
            "b": b,
            "E": E if f is not None else None,
            "f": f,
            "iters": 50,
            "tau": 0.05
        }
    else:
        cons = None

    return aff, node_aff_np, edge_aff_np, n1_center, n2_center, cons


def qap_gm(K, n1, n2, method="rrwm", cons=None):
    if method == "rrwm":
        soft = pygm.rrwm(K, n1, n2)
        if cons is not None:
            soft_cons = linsat_layer(soft.contiguous().view(-1), A=cons["A"], b=cons["b"], E=cons["E"], f=cons["f"],
                                     max_iter=cons["iters"], tau=cons["tau"]).reshape(n1, n2)
            X = pygm.hungarian(soft_cons).detach().numpy()
            X_soft = soft_cons.detach().numpy()
        else:
            X = pygm.hungarian(soft).detach().numpy()
            X_soft = soft.detach().numpy()
    else:
        raise ValueError("method should be one of 'rrwm', 'ipfp', 'sm', 'ngm'")

    return {"X": X, "X_soft": X_soft, "X_filter": []}


def gm_iterative(G_q, G_db, dataset_info, iters=0, lower=0.5,
                 ransacReprojThreshold=10, confidence=0.99, maxIters_for_ransac=100,
                 methods=None, vis=False, match=None, cons_flag=False):
    methods = ["rrwm"]
    aff0, node_aff, edge_aff, node1_center, node2_center, cons = build_affinity_mat(
        G_q, G_db, dataset_info, cons_flag=cons_flag, vis=vis, match=match)
    inlier_num = {k: [] for k in methods}
    transformations = {k: [] for k in methods}
    scores = {k: [] for k in methods}
    if aff0 is None:
        match_results = {k: None for k in methods}
        affs = {k: None for k in methods}
        scores = {k: [0] for k in methods}
    else:
        match_results = {}
        affs = {}
        for method in methods:
            match_results[method] = []
            affs[method] = []
            aff = aff0.clone()
            affs[method].append(aff0)

            qap_result = qap_gm(aff, node1_center.shape[0], node2_center.shape[0], method, cons)
            score = compute_score(qap_result["X"], aff0, node_aff)
            scores[method].append(score)
            inlier_num[method] = [-10]
            transformations[method] = [None]
            match_results[method].append(qap_result.copy())
            affs[method].append(aff.clone())

            if iters != 0:
                for i in range(iters):
                    src_kps = []
                    dst_kps = []
                    node_q, node_db = np.where(qap_result["X"] == 1)
                    for node_i in range(node_q.size):
                        src_kps.append(node1_center[node_q[node_i]])
                        dst_kps.append(node2_center[node_db[node_i]])
                    if len(src_kps) >= 3:
                        transform, mask = cv2.estimateAffinePartial2D(np.array(src_kps), np.array(dst_kps),
                                                                      method=cv2.RANSAC,
                                                                      ransacReprojThreshold=ransacReprojThreshold,
                                                                      confidence=confidence,
                                                                      maxIters=maxIters_for_ransac)
                        inlier_prop = mask.sum()
                        if inlier_prop < 0:
                            print("inlier_prop is less than 0")
                        X_ransac_filter = []
                        for k in range(node_q.size):
                            index = node_q[k] + node_db[k] * qap_result["X"].shape[0]
                            if mask[k] == 0:
                                qap_result["X"][node_q[k], node_db[k]] = 0
                                node_sim = aff[index, index]
                                if node_sim > 0:
                                    aff[index, index] = node_sim * lower
                                else:
                                    aff[index, index] = node_sim / lower
                                X_ransac_filter.append((node_q[k], node_db[k]))
                            else:
                                aff[index, index] = aff[index, index]
                            qap_result["X_filter"] = X_ransac_filter
                    else:
                        inlier_prop = -(len(src_kps))
                        transform = None
                    qap_result = qap_gm(aff, node1_center.shape[0], node2_center.shape[0], method, cons)
                    score = compute_score(qap_result["X"], aff0, node_aff)
                    scores[method].append(score)
                    inlier_num[method].append(inlier_prop)
                    transformations[method].append(transform)
                    match_results[method].append(qap_result.copy())
                    affs[method].append(aff.clone())
    match_results.update({"node_aff": node_aff, "edge_aff": edge_aff})
    return match_results, scores, inlier_num, affs, transformations
