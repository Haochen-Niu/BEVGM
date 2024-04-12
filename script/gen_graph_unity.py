# -*-coding:utf-8-*-
import os
import argparse
import yaml
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx
import sys
import gc

sys.path.insert(0, os.getcwd())
from core.extract_topo import fv_gen_topo
from dataset.dataloader import SynthiaDataLoader
from models.my_test import load_mrNVLAD_model

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('--config', default='config/gm.yaml', help='config file path')
    parser.add_argument('--vis_notsave', action='store_true', help='save the vis or not')
    parser.add_argument('--dataset_type', default='SYNTHIA', help='dataset type')
    parser.add_argument('--dataset_path', default='/home/nhc/16T/dataset/SYNTHIA/SYNTHIA_VIDEO_SEQUENCES',
    					help='dataset path')
    parser.add_argument('--dataset_name', nargs='+', type=str, default=["dawnF", "fallB"])
    parser.add_argument('--graph_dir', default='output/graph_test', help='pair file path')
    parser.add_argument("--wm", type=int, help="output image width in [m]", default=80)
    parser.add_argument("--hm", type=int, help="output image height in [m], (depth)", default=40)
    parser.add_argument("--r", type=int, help="output image resolution in [px/m]", default=20)
    parser.add_argument("--thread", type=int, help="thread number", default=1)

    args = parser.parse_args()
    return args


def parallel_gen_graph(data_loader, timestamp, metadata, graph_path, match=None):
    graph_save_path = os.path.join(graph_path, timestamp + ".gpickle")
    if os.path.exists(graph_save_path):
        return 0
    else:
        fv = data_loader.load_rgb(timestamp)
        depth = data_loader.load_depth(timestamp)
        # seg = data_loader.load_seg(timestamp)
        seg = data_loader.load_seg_gt(timestamp)
        seg_vis = data_loader.load_seg_vis(timestamp)
        if match is not None:
            appearance = match
        else:
            appearance = None

        _, angle = data_loader.get_cam_pose(timestamp)
        data_loader.update_ipm(angle)

        graph = fv_gen_topo(fv=fv, depth=depth, seg=seg, appearance=appearance, ipm_info=data_loader.ipm_info,
                            metadata=metadata, depth_thr=data_loader.hm, seg_vis=seg_vis, vis=False)

        nx.write_gpickle(graph, graph_save_path)

        gc.collect()
        return 0


def main():
    args = parse_args()
    args_dict = vars(args)
    with open(args.config, 'r') as file:
        data = file.read()
        cfg = yaml.load(data, Loader=yaml.FullLoader)
    metadata = cfg["data_dic"]['fv']

    dataset_names = args.dataset_name
    dataset_path = args.dataset_path
    dataset_type = args.dataset_type
    match = load_mrNVLAD_model()

    for dataset in dataset_names:
        if dataset_type == "SYNTHIA":
            data_loader = SynthiaDataLoader()
        else:
            raise NotImplementedError
        data_loader.get_dataset_info(dataset_path, dataset)
        data_loader.get_ipm_init(args.r, args.wm, args.hm)  # get ipm info
        timestamp_list = data_loader.get_timestamp_info()

        graph_path = os.path.join(args.graph_dir,
                                  args.dataset_type + dataset + "_graph_" + str(args.wm) + "_" + str(args.hm))
        vis_path = os.path.join(args.graph_dir,
                                args.dataset_type + dataset + "_vis_" + str(args.wm) + "_" + str(args.hm))
        if not os.path.exists(graph_path) or not os.path.exists(vis_path):
            os.makedirs(graph_path, exist_ok=True)
            os.makedirs(vis_path, exist_ok=True)

        with open(os.path.join(graph_path, "args.json"), "w") as fw:
            args_json = json.dumps(args_dict, indent=4, ensure_ascii=False)
            fw.write(args_json)
        fw.close()
        print("parameters saved in args.json")

        results = (
            Parallel(n_jobs=args.thread)(
                delayed(parallel_gen_graph)
                (data_loader, timestamp, metadata, graph_path, match)
                for timestamp in tqdm(timestamp_list))
        )
        print("finished one of dataset_names: {}".format(dataset))


if __name__ == '__main__':
    main()
