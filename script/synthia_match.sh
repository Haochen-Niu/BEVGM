#!/bin/bash
params_e=($1)
params_graph="output/"$2
params_match="output/"$3
exps=("SYNTHIA_springB_dawnF" "SYNTHIA_summerB_dawnF" "SYNTHIA_fallB_dawnF" \
      "SYNTHIA_winterB_dawnF" "SYNTHIA_sunsetB_dawnF" "SYNTHIA_nightB_dawnF" )
params_iters=1
params_wh=("80_40")
params_n=(50)
params_thread=(1)
params_coarse=("mrNVLAD")
default_params_vis_save=(0)
params_vis_save=${4:-$default_params_vis_save}
params_reproj=(200)
params_cons=(1)
params_lowers=(0.9)

for param_e in "${params_e[@]}"; do
  for param_cons in "${params_cons[@]}"; do
    for param_lower in "${params_lowers[@]}"; do
      python -W ignore script/match_unity.py \
        --config config/gm_synthia.yaml \
        --wh "$params_wh" \
        --topN "$params_n" \
        --sample 1 1 \
        --exp ${exps[$param_e]} \
        --thread "$params_thread" \
        --coarse "$params_coarse" \
        --vis_save "$params_vis_save" \
        --reproj "$params_reproj" \
        --iters "$params_iters" \
        --graph "$params_graph" \
        --match_dir "$params_match" \
        --cons "$param_cons" \
        --lower "$param_lower"

      echo "*********************finished!*********************"
      echo ""
    done
  done
done