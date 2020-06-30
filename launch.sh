#!/bin/bash

LOGGING_DIR = $1

CUDA_VISIBLE_DEVICES="0" python train.py --alg TD3 --env Ant-v2 --logging_dir $LOGGING_DIR/td3_ant_0
CUDA_VISIBLE_DEVICES="0" python train.py --alg TD3 --env Ant-v2 --logging_dir $LOGGING_DIR/td3_ant_1
CUDA_VISIBLE_DEVICES="0" python train.py --alg TD3 --env Ant-v2 --logging_dir $LOGGING_DIR/td3_ant_2

CUDA_VISIBLE_DEVICES="1" python train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/td3_cheetah_0
CUDA_VISIBLE_DEVICES="1" python train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/td3_cheetah_1
CUDA_VISIBLE_DEVICES="1" python train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/td3_cheetah_2

CUDA_VISIBLE_DEVICES="2" python train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGGING_DIR/td3_humanoid_0
CUDA_VISIBLE_DEVICES="2" python train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGGING_DIR/td3_humanoid_1
CUDA_VISIBLE_DEVICES="2" python train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGGING_DIR/td3_humanoid_2

CUDA_VISIBLE_DEVICES="3" python train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGGING_DIR/td3_walker2d_0
CUDA_VISIBLE_DEVICES="3" python train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGGING_DIR/td3_walker2d_1
CUDA_VISIBLE_DEVICES="3" python train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGGING_DIR/td3_walker2d_2

CUDA_VISIBLE_DEVICES="4" python train.py --alg SAC --env Ant-v2 --logging_dir $LOGGING_DIR/sac_ant_0
CUDA_VISIBLE_DEVICES="4" python train.py --alg SAC --env Ant-v2 --logging_dir $LOGGING_DIR/sac_ant_1
CUDA_VISIBLE_DEVICES="4" python train.py --alg SAC --env Ant-v2 --logging_dir $LOGGING_DIR/sac_ant_2

CUDA_VISIBLE_DEVICES="5" python train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/sac_cheetah_0
CUDA_VISIBLE_DEVICES="5" python train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/sac_cheetah_1
CUDA_VISIBLE_DEVICES="5" python train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGGING_DIR/sac_cheetah_2

CUDA_VISIBLE_DEVICES="6" python train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGGING_DIR/sac_humanoid_0
CUDA_VISIBLE_DEVICES="6" python train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGGING_DIR/sac_humanoid_1
CUDA_VISIBLE_DEVICES="6" python train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGGING_DIR/sac_humanoid_2

CUDA_VISIBLE_DEVICES="7" python train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGGING_DIR/sac_walker2d_0
CUDA_VISIBLE_DEVICES="7" python train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGGING_DIR/sac_walker2d_1
CUDA_VISIBLE_DEVICES="7" python train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGGING_DIR/sac_walker2d_2
