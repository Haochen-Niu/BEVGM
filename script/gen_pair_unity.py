# -*-coding:utf-8-*-
import argparse
import os
from tqdm import tqdm
import numpy as np
import sys
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.getcwd())
from dataset.dataloader import SynthiaDataLoader
from tools.utils import cal_comvis_area


def gen_pairs(dataset_path, dataset_name1, dataset_name2, dataset_type="airsim", output_path="./", ):
    if dataset_type == "SYNTHIA":
        data_loader_q = SynthiaDataLoader()
        data_loader_q.get_dataset_info(dataset_path, dataset_name1)
        data_loader_db = SynthiaDataLoader()
        data_loader_db.get_dataset_info(dataset_path, dataset_name2)
        fov = 100
        ranger_x = 12
        ranger_y = 40
    else:
        raise NotImplementedError

    euler_order = 'YXZ'

    timestamps_q = data_loader_q.get_timestamp_info()
    timestamps_db = data_loader_db.get_timestamp_info()
    ts1_pair = {}
    common_areas = []
    for timestamp_q in tqdm(timestamps_q):
        Twc_q, _ = data_loader_q.get_cam_pose(timestamp_q)
        Twc_q_inv = np.linalg.inv(Twc_q)
        pairs_for_q = {}
        for timestamp_db in timestamps_db:
            Twc_db, _ = data_loader_db.get_cam_pose(timestamp_db)
            distance = np.linalg.norm(Twc_q[:3, 3] - Twc_db[:3, 3])
            Tq_db = Twc_q_inv @ Twc_db
            angle = Rotation.from_matrix(Tq_db[:3, :3]).as_euler(euler_order, degrees=True)
            if distance > 2 * (ranger_x + ranger_y):
                common_vis_area = 0
            else:
                common_vis_area = cal_comvis_area(Tq_db[[0, 2], 3], angle[0],
                                                  ranger_x=ranger_x, ranger_y=ranger_y, fov=fov, vis=False)
            pairs_for_q[str(timestamp_db)] = {
                "distance": distance,
                "T": Twc_db,
                "delta_T": Tq_db,
                "angle": angle,
                "com_area": common_vis_area,
            }
            common_areas.append(common_vis_area) if common_vis_area > 0 else None

        ts1_pair[str(timestamp_q)] = {"timestamp": str(timestamp_q),
                                      "T": Twc_q,
                                      "pairs_with_q": pairs_for_q,
                                      }
    ts1_pair.update({
        "query_list": sorted(timestamps_q),
        "db_list": sorted(timestamps_db)})

    np.save(output_path, ts1_pair)
    print(output_path, "saved!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default="SYNTHIA")
    parser.add_argument("--dataset_path", "-p", type=str,
                        default="/home/nhc/16T/dataset/SYNTHIA/SYNTHIA_VIDEO_SEQUENCES")
    parser.add_argument("--dataset_name1", "-n1", type=str, default="springB")
    parser.add_argument("--dataset_name2", "-n2", type=str, default="dawnF")

    parser.add_argument("--output_path", "-o", type=str, default="script/pairs")
    args = parser.parse_args()

    pair_type = args.type + "_" + args.dataset_name1 + "_" + args.dataset_name2 + "_pairs.npy"
    output_path = os.path.join(args.output_path, pair_type)
    gen_pairs(args.dataset_path, args.dataset_name1, args.dataset_name2, args.type, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
