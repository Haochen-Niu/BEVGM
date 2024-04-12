# -*-coding:utf-8-*-
import argparse
import os

import re
import faiss
import numpy as np
import math
import time
import sys

sys.path.insert(0, os.getcwd())
from tools.utils import compute_PR


def get_gt(dis, angle, area_prop, area_thr):
    if abs(angle) > 150:
        if area_prop > area_thr:
            return 2
        else:
            return 0
    elif abs(angle) < 30:
        if dis < 10:
            return 1
        else:
            return 0
    elif area_prop > area_thr:
        return 3
    else:
        return 0


def get_eval(index_timestamp_q, index_timestamp_db, pairs_dict,
             matches_index, scores, topN, output_path,
             area_thr=0.5):
    pred = []
    gt = []
    gt_direction = []
    same_dir = 0
    opposite_dir = 0
    LCD_dis = []

    querys_T = []
    dbs_T = []
    for i in range(len(index_timestamp_q)):
        q_i_timestamp = index_timestamp_q[i]
        db_best_index = matches_index[i, 0]
        db_best_timestamp = index_timestamp_db[db_best_index]
        score = scores[i, 0]
        pred.append(score)
        dis = pairs_dict[q_i_timestamp]["pairs_with_q"][db_best_timestamp]["distance"]
        delta_T = pairs_dict[q_i_timestamp]["pairs_with_q"][db_best_timestamp]["delta_T"]
        angle = pairs_dict[q_i_timestamp]["pairs_with_q"][db_best_timestamp]["angle"][0]
        querys_T.append(pairs_dict[q_i_timestamp]["T"])
        dbs_T.append(pairs_dict[q_i_timestamp]["pairs_with_q"][db_best_timestamp]["T"])
        com_area = pairs_dict[q_i_timestamp]["pairs_with_q"][db_best_timestamp]["com_area"]
        groundtruth = get_gt(dis, angle, area_prop=com_area, area_thr=area_thr)
        gt_direction.append(groundtruth)
        gt.append(1 if groundtruth else 0)
        LCD_dis.append(dis)
        if groundtruth == 1:  # easy sample
            same_dir += 1
        elif groundtruth == 2 or groundtruth == 3:  # hard sample
            opposite_dir += 1
    F1_score, precision, recall, auc, pr_thresholds, precision_r100, recall_p100, right_ids_p100 = \
        compute_PR(gt, pred)
    same_dir_100p = np.count_nonzero(np.array(gt_direction)[right_ids_p100] == 1) \
        if right_ids_p100 is not None else 0
    opposite_dir_100p = (np.count_nonzero(np.array(gt_direction)[right_ids_p100] == 2) +
                         np.count_nonzero(np.array(gt_direction)[right_ids_p100] == 3)) \
        if right_ids_p100 is not None else 0
    result = {"F1_score": F1_score, "precision": precision, "recall": recall, "auc": auc, "gt": gt,
              "data_path": output_path, "querys_T": querys_T, "dbs_T": dbs_T,
              "same": same_dir, "opposite": opposite_dir, "recall100": recall_p100, "precision100": precision_r100,
              "same100": same_dir_100p, "opposite100": opposite_dir_100p}
    print("F1_score: {} \n"
          "auc: {} \n"
          "same: {} \n"
          "opposite: {} \n"
          "same@100: {} \n"
          "opposite@100: {} \n"
          "recall@100: {} \n"
          "precision100: {}".format(F1_score, auc, same_dir, opposite_dir, same_dir_100p,
                                    opposite_dir_100p, recall_p100, precision_r100))
    return result


def eval_vec_method(topN=10, exp=None, mode=0, overwrite=False, area_thr=0.5, method="mrNVLAD"):
    data_path = os.path.join("output/compare", method)

    if not os.path.exists(os.path.join(data_path, exp)):
        os.makedirs(os.path.join(data_path, exp))

    result_path = os.path.join(data_path, exp, str(mode) + "_eval_result.npy")
    if os.path.exists(result_path) and not overwrite:
        result = np.load(result_path, allow_pickle=True).item()
        if mode == 0:
            print("F1_score: {} \n"
                  "auc: {} \n"
                  "same: {} \n"
                  "opposite: {} \n"
                  "same@100: {} \n"
                  "opposite@100: {} \n"
                  "recall@100: {} \n"
                  "precision100: {}".format(result['F1_score'], result['auc'], result['same'], result['opposite'],
                                            result['same100'], result['opposite100'],
                                            result["recall100"], result["precision100"]))
        return result

    coarse_matches_path = os.path.join(data_path, exp, method + "_coarse_matches.npy")
    if os.path.exists(coarse_matches_path) and not overwrite:
        data = np.load(coarse_matches_path, allow_pickle=True).item()
        index_timestamp_q = data["query_list"]
        index_timestamp_db = data["database_list"]
        scores = data["scores"]
        matches_index = data["matches_index"]
    else:
        datatype = exp.split("_")[0]
        dataset1 = exp.split("_")[1]
        dataset2 = exp.split("_")[2]

        qFeat_dic = np.load(os.path.join(data_path, datatype + "_" + dataset1 + "_Feat.npy"), allow_pickle=True).item()
        dbFeat_dic = np.load(os.path.join(data_path, datatype + "_" + dataset2 + "_Feat.npy"), allow_pickle=True).item()

        Feat_q = qFeat_dic['feat']
        index_timestamp_q = [re.split("[_.]", os.path.basename(timestamp))[0] for timestamp in qFeat_dic["list"]]

        Feat_db = dbFeat_dic['feat']
        index_timestamp_db = [re.split("[_.]", os.path.basename(timestamp))[0] for timestamp in dbFeat_dic["list"]]

        feat_size = Feat_db.shape[1]
        faiss_index = faiss.IndexFlatL2(feat_size)
        faiss_index.add(Feat_db)

        search_num = len(index_timestamp_db)

        distance_matrix, matches_index = faiss_index.search(Feat_q, search_num)
        scores = 2 - distance_matrix

        coarse_matches = {
            "query_list": index_timestamp_q,
            "database_list": index_timestamp_db,
            "scores": scores,
            "matches_index": matches_index
        }
        np.save(coarse_matches_path, coarse_matches)
        print("output coarse matches for bevtopo")

    pairs_path = os.path.join("script/pairs", exp + "_pairs.npy")
    pairs_dict = np.load(pairs_path, allow_pickle=True).item()
    output_path = os.path.join(data_path, exp)
    if mode == 0:
        topN = 0
    result = get_eval(index_timestamp_q, index_timestamp_db, pairs_dict, matches_index, scores, topN, output_path, area_thr=area_thr)
    result.update({"method": method})
    np.save(result_path, result)
    return result


def get_bev_scores(topN=10, exp=None, iters=-1, data_path=None):
    method = "rrwm"
    output_path = os.path.join(data_path, exp)
    bevtopo_results = np.load(os.path.join(output_path, "match_result.npy"), allow_pickle=True).item()
    index_timestamps_db = bevtopo_results["dataset_db_list"]
    match_result = bevtopo_results["match_result"]
    index_timestamps_q = []
    scores = np.zeros((len(match_result), topN))
    matches_index = np.zeros_like(scores, dtype=np.int64)

    coarse_score_list = []
    bow_wrong1 = 0
    bow_wrong2 = 0

    time_cost = []
    for i in range(len(match_result)):
        i_scores = []
        bev_scores = []
        query = match_result[i]
        index_timestamps_q.append(query["timestamp_q"])
        match_info = query["match_info"]
        bow_q = query["bow_q"]
        bow_q_building = bow_q[2]
        self_score_q = query["self_score"]["rrwm"][-1]
        db_order_in_match_info = query["matches_index"]

        start_time = time.time()
        for j in range(topN):
            q_db_info = match_info[j]
            if len(q_db_info["inlier"][method]):
                bevtopo_score = q_db_info["scores"][method][iters]
                inlier_num = q_db_info["inlier"][method][iters]
                transformation = q_db_info["transformations"][method][iters]
                angle = math.atan2(transformation[1, 0],
                                   transformation[0, 0]) * 180 / math.pi if transformation is not None else None
                X_soft = q_db_info["match"][method][iters]["X_soft"]
                if np.sum(X_soft) == 0:
                    inlier_num = -20
                else:
                    num = 0
                    entropy = 0
                    for value in X_soft.flat:
                        if value < 10e-6:
                            continue
                        elif value == 1:
                            value = 1 - 10e-6
                        num += 1
                        entropy += -value * np.log2(value) - (1 - value) * np.log2(1 - value)
                    if num == 0:
                        entropy = 1
                    else:
                        entropy /= (num * 1)
            else:
                inlier_num = -20
                angle = None
            coarse_score = q_db_info["coarse_score"]
            bow_db = q_db_info["bow_db"]
            bow_db_building = bow_db[2]

            if sum(bow_q) == 0 or bow_q[2] == 0 or bow_q_building != bow_db_building:
                inlier_num = -15
                sim_bow = 0
                bow_wrong1 += 1
            else:
                sim_bow = sum(list(map(lambda x: abs(x[0] - x[1]), zip(bow_q, bow_db)))) / sum(bow_q)
            if sim_bow > 1.5:
                inlier_num = -10
                bow_wrong2 += 1

            # add angle info
            if angle is None:
                coarse_factor = 1
            elif 30 < abs(angle) < 150:
                coarse_factor = 0.8
            else:
                coarse_factor = 1

            if iters == 0:
                inlier_num = 1
            if inlier_num < -5:
                final_score = 10e-6
                bev_score = 10e-6
            else:
                bevtopo_score_norm = bevtopo_score / self_score_q if self_score_q else 0.01
                final_score = (1 + bevtopo_score_norm * (1 - entropy)) * (
                        1 + coarse_score * entropy) * coarse_factor
                bev_score = bevtopo_score_norm
                coarse_score_list.append(final_score)

            i_scores.append(final_score)
            bev_scores.append(bev_score)

        i_order = np.argsort(bev_scores)[::-1]
        scores[i, :] = np.array(i_scores)[i_order][:topN]
        matches_index[i, :] = db_order_in_match_info[i_order][:topN]
        time_cost.append(time.time() - start_time)

    return index_timestamps_q, index_timestamps_db, matches_index, scores


def eval_bevtopo(topN=10, exp=None, mode=0, iters=-1, data_path=None,area_thr=0.5):
    index_timestamps_q, index_timestamps_db, matches_index, scores = get_bev_scores(
        topN, exp, iters, data_path)
    pairs_path = os.path.join("script/pairs", exp + "_pairs.npy")
    pairs_dict = np.load(pairs_path, allow_pickle=True).item()
    if mode == 0:
        topN = 0
    result = get_eval(index_timestamps_q, index_timestamps_db, pairs_dict, matches_index, scores, topN, data_path,
                      area_thr=area_thr,
                      )
    result.update({"method": "bevtopo"})

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bev_datapath', "-p", default=None)
    parser.add_argument('--exp', "-e", default=None)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('--topN', '-n', type=int, default=50)
    parser.add_argument('--area_thr', type=float, default=0.3)
    parser.add_argument('--method', type=str, default="rrwm")
    parser.add_argument('--mode', type=str, default="pr")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    topN = args.topN
    area_thr = args.area_thr
    exp = args.exp
    bev_datapath = args.bev_datapath
    mode = 0 if args.mode == "pr" else 1  # 0 means PR, else means topN

    results = []

    print("---------------------------mrNVLAD---------------------------")
    result = eval_vec_method(topN, exp, mode,
                             area_thr=area_thr)
    results.append(result)

    print("---------------------------bevtopo---------------------------")
    result = eval_bevtopo(topN, exp, mode,
                          iters=args.iters, data_path=bev_datapath, area_thr=area_thr,
                          )
    results.append(result)


if __name__ == '__main__':
    main()
