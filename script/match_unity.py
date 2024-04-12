# -*-coding:utf-8-*-
import json
import networkx as nx
import numpy as np
import yaml
import argparse
import os
from tqdm import tqdm
import pygmtools as pygm
import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.getcwd())

from core.topo_match import gm_iterative

pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)


def load_q_db(data_q, args):
    coarse_path = "./output/compare/"
    method = args.coarse
    if method == str(0):
        dataset_q_list = data_q["query_list"][::args.sample[0]]
        dataset_db_list = data_q["db_list"]
        matches_index = np.tile(np.arange(0, len(dataset_db_list), args.sample[1]), (len(dataset_q_list), 1))
        coarse_scores = np.ones_like(matches_index, dtype=np.float32)
        print("--------------only bevtopo------------------")
    else:
        coarse_matches = np.load(
            os.path.join(coarse_path, method, args.exp, method + "_coarse_matches.npy"), allow_pickle=True).item()
        dataset_q_list = coarse_matches["query_list"][::args.sample[0]]
        dataset_db_list = coarse_matches["database_list"]
        matches_index = coarse_matches["matches_index"][::args.sample[0], :args.topN]
        coarse_scores = coarse_matches["scores"][::args.sample[0], :args.topN]
        print("--------------{} based------------------".format(method))

    db_need_load = list(set(matches_index.flatten()))

    return dataset_q_list, dataset_db_list, matches_index, coarse_scores, db_need_load


def topo_match_for_pairs(query, graph_q_dict, graphs_db, dataset_db_list, matches_index, coarse_scores,
                         dataset_info, match_path, args):
    match_info = []
    T_q = query["T"]
    timestamp_q = graph_q_dict["timestamp_q"]
    graph_q = graph_q_dict["graph"]
    self_scores_q = graph_q.graph["self_scores"]
    save_path = os.path.join(match_path, timestamp_q + ".npy")
    if os.path.exists(save_path):
        return 0

    for i in range(len(matches_index)):
        db_index = matches_index[i]
        timestamp_db = dataset_db_list[db_index]
        coarse_score = coarse_scores[i]
        graph_db = graphs_db[timestamp_db]["graph"]
        match_results, scores, inlier, affs, transformations = gm_iterative(graph_q, graph_db, dataset_info,
                                                                            iters=args.iters,
                                                                            lower=args.lower,
                                                                            ransacReprojThreshold=args.reproj,
                                                                            confidence=0.9,
                                                                            maxIters_for_ransac=args.ransac,
                                                                            methods=args.methods,
                                                                            cons_flag=args.cons)

        # stage: calculate pose
        # todo  include:transformations,pose1,angle1,pose2,angle2

        # stage: group the info
        timestamp_db_info = {
            "timestamp_db": timestamp_db,
            "match": match_results,
            "inlier": inlier,
            "transformations": transformations,
            "scores": scores,
            "bow_db": graph_db.graph["bow"],
            "coarse_score": coarse_score,
        }
        match_info.append(timestamp_db_info)
        plt.close('all')

    match_result = {
        "timestamp_q": timestamp_q,
        "T_q": T_q,
        "bow_q": graph_q.graph["bow"],
        "match_info": match_info,
        "self_score": self_scores_q,
        "matches_index": matches_index,
    }
    # np.save(save_path, match_result)
    return match_result


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('--config', default='config/gm.yaml', help='config file path')
    parser.add_argument('--vis', default=True, action='store_true', help='visualize the input')
    parser.add_argument('--methods', default=["rrwm"])
    parser.add_argument('--vis_save', type=int, default=1, help='save the visualization result')
    parser.add_argument('--exp', help='exp name')
    parser.add_argument('--pair_path', default='script/pairs', help='pair file path')
    parser.add_argument('--graph_dir', help='graph file path')
    parser.add_argument("--wh", type=str, default='80_40')
    parser.add_argument('--match_dir', help='output path')
    parser.add_argument('--coarse', type=str, default='mrNVLAD', help='coarse match method')
    parser.add_argument('--topN', type=int, default=20, help='coarse match threshold')
    parser.add_argument('--sample', nargs='+', type=int, default=[1, 1], help='sample interval')
    parser.add_argument('--iters', type=int, default=3, help='gm iteration times')
    parser.add_argument('--reproj', type=int, default=40, help='ransca reproj threshold')
    parser.add_argument('--ransac', type=int, default=100, help='ransca iteration times')
    parser.add_argument('--lower', type=float, default=0.9, help='lower the similarity between wrong match')
    parser.add_argument("--thread", type=int, help="thread number", default=1)
    parser.add_argument("--cons", type=int, help='add constrains or not', default=1)

    args = parser.parse_args()
    args.cons = True if args.cons == 1 else False
    return args


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        data = file.read()
        cfg = yaml.load(data, Loader=yaml.FullLoader)
    data_type = "fv"
    dataset_info = cfg["data_dic"][data_type]
    args.vis = False
    dataset_type = os.path.basename(args.exp).split('_')[0]
    dataset_q_name = os.path.basename(args.exp).split('_')[1]
    dataset_db_name = os.path.basename(args.exp).split('_')[2]

    data_q = np.load(os.path.join(args.pair_path, args.exp + "_pairs.npy"), allow_pickle=True).item()
    dataset_q_list, dataset_db_list, matches_index, coarse_scores, db_need_load = load_q_db(data_q, args)
    match_path = os.path.join(
        args.match_dir + str(args.sample).replace(" ", "") + "_wh" + args.wh +
        "_gm" + str(args.iters) + "_rproj" + str(args.reproj) + "_top" + str(
            args.topN) + "_" + args.coarse + "_" + str(args.lower), args.exp)
    if not os.path.exists(match_path):
        os.makedirs(match_path)
    graphs_q = {}
    for q_i in tqdm(range(len(dataset_q_list))):
        q = dataset_q_list[q_i]
        graph_q_path = os.path.join(args.graph_dir, dataset_type + dataset_q_name + "_graph_" + args.wh)
        graph_q_save_path = os.path.join(graph_q_path, q + ".gpickle")
        if os.path.exists(graph_q_save_path):
            graph = nx.read_gpickle(graph_q_save_path)
            graphs_q[q] = {
                "timestamp_q": q,
                "graph": graph,
                "q_i": q_i,
            }
        else:
            print("graphs_q file not exist: {}".format(graph_q_save_path))
            exit()

    graphs_db = {}
    for db_i in tqdm(db_need_load):
        db = dataset_db_list[db_i]
        graph_db_path = os.path.join(args.graph_dir, dataset_type + dataset_db_name + "_graph_" + args.wh)
        graph_db_save_path = os.path.join(graph_db_path, db + ".gpickle")
        if os.path.exists(graph_db_save_path):
            graph = nx.read_gpickle(graph_db_save_path)
            graphs_db[db] = {
                "timestamp_db": db,
                "graph": graph,
                "db_i": db_i,
            }
        else:
            print("graphs_db file not exist: {}".format(graph_db_save_path))
            exit()

    print("data size need to process:{}".format(matches_index.shape))

    args_dict = vars(args)
    with open(os.path.join(match_path, "args.json"), "w") as fw:
        args_json = json.dumps(args_dict, indent=4, ensure_ascii=False)
        fw.write(args_json)
    # fw.write('\n')
    fw.close()
    print("parameters saved in args.json")
    match_results = (
        Parallel(n_jobs=args.thread)(
            delayed(topo_match_for_pairs)
            (data_q[dataset_q_list[query_index]], graphs_q[dataset_q_list[query_index]], graphs_db,
             dataset_db_list, matches_index[query_index, :], coarse_scores[query_index, :],
             dataset_info, match_path, args)
            for query_index in tqdm(range(len(dataset_q_list))))
    )
    print("save result")
    save_path = os.path.join(match_path, "match_result.npy")
    results = {
        "match_result": match_results,
        "dataset_db_list": dataset_db_list,
    }
    np.save(save_path, results)


if __name__ == '__main__':
    main()
