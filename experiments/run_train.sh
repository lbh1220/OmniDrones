#!/bin/bash

set -euo pipefail

# 通用训练参数
num_envs=512
n_steps=256
num_mini_batch=64
total_timesteps=100000000
evtol_radius=8.0
drones_num=10
evtols_num=1
# # baseline
# python learning/train_traffic_withtest.py \
#     --num_envs $num_envs \
#     --n_steps $n_steps \
#     --num_mini_batch $num_mini_batch \
#     --total_timesteps $total_timesteps \
#     --drones_num 10 \
#     --evtols_num 1 \
#     --evtol_future_penalty 0.0 \
#     --drone_future_penalty 0.0 \
#     --norm_reward \
#     --use_global_path \
#     --video \
#     --rew_cross_track_coeff -0.1 \
#     --action_space_type "discrete" \
#     --predict_steps 0 \
#     --experiment_name "ablation/u10e1/baseline"

drone_future_penalty=-2.0
evtol_future_penalty_list=(-2.0 -0.0 -4.0 -8.0)
for evtol_future_penalty in ${evtol_future_penalty_list[@]}; do
    python learning/train_traffic_withtest.py \
        --num_envs $num_envs \
        --n_steps $n_steps \
        --num_mini_batch $num_mini_batch \
        --total_timesteps $total_timesteps \
        --drones_num $drones_num \
        --evtols_num $evtols_num \
        --evtol_radius $evtol_radius \
        --evtol_future_penalty $evtol_future_penalty \
        --drone_future_penalty $drone_future_penalty \
        --norm_reward \
        --use_global_path \
        --rew_cross_track_coeff -0.1 \
        --action_space_type "gaussian" \
        --predict_steps 5 \
        --evtols_decay_factor 1.0 \
        --evtols_threshold_factor 1.5 \
        --use_angle_distance_obs \
        --use_type_split_attn \
        --experiment_name "future_penalty/u${drones_num}e${evtols_num}_r${evtol_radius}/split_attn/fu${drone_future_penalty}fe${evtol_future_penalty}"
done
