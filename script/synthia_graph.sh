#!/bin/bash
gpu=$1
params_e=($2)
params_graph=$3
dataset_path=$4
exps=("dawnF" "springB" "summerB" "fallB" "winterB" "sunsetB" "nightB")
params_hm=40
params_wm=80
params_r=20
default_thread=3
params_thread=${5:-$default_thread}
params_type="SYNTHIA"
dataset_path="/home/nhc/16T/dataset/SYNTHIA/SYNTHIA_VIDEO_SEQUENCES"
params_vis_save=0

for param_p in "${params_graph[@]}"; do
  exp=()
  for param_e in "${params_e[@]}"; do
    exp+="${exps[$param_e]} "
  done
  echo "exp: ${exp[@]}"
  echo "************************start************************"
  CUDA_VISIBLE_DEVICES=${gpu} python script/gen_graph_unity.py \
      --config config/gm_synthia.yaml \
      --graph_dir $params_graph \
      --thread $params_thread \
      --hm $params_hm \
      --wm $params_wm \
      --r $params_r \
      --dataset_type $params_type \
      --dataset_path $dataset_path \
      --dataset_name ${exp} \
      --vis_notsave
  echo "************************finished!************************"
done