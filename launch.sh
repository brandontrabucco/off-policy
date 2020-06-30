#!/bin/bash

PKGDIR=/packages
LOGDIR=/global/shared/btrabucco

. $PKGDIR/anaconda3/etc/profile.d/conda.sh

conda activate offpolicy

CUDA_VISIBLE_DEVICES="0" python $PKGDIR/off-policy/train.py --alg TD3 --env Ant-v2 --logging_dir $LOGDIR/td3_ant_0 &
pids[0]=$!
CUDA_VISIBLE_DEVICES="0" python $PKGDIR/off-policy/train.py --alg TD3 --env Ant-v2 --logging_dir $LOGDIR/td3_ant_1 &
pids[1]=$!
CUDA_VISIBLE_DEVICES="0" python $PKGDIR/off-policy/train.py --alg TD3 --env Ant-v2 --logging_dir $LOGDIR/td3_ant_2 &
pids[2]=$!

CUDA_VISIBLE_DEVICES="1" python $PKGDIR/off-policy/train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGDIR/td3_cheetah_0 &
pids[3]=$!
CUDA_VISIBLE_DEVICES="1" python $PKGDIR/off-policy/train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGDIR/td3_cheetah_1 &
pids[4]=$!
CUDA_VISIBLE_DEVICES="1" python $PKGDIR/off-policy/train.py --alg TD3 --env HalfCheetah-v2 --logging_dir $LOGDIR/td3_cheetah_2 &
pids[5]=$!

CUDA_VISIBLE_DEVICES="2" python $PKGDIR/off-policy/train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGDIR/td3_humanoid_0 &
pids[6]=$!
CUDA_VISIBLE_DEVICES="2" python $PKGDIR/off-policy/train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGDIR/td3_humanoid_1 &
pids[7]=$!
CUDA_VISIBLE_DEVICES="2" python $PKGDIR/off-policy/train.py --alg TD3 --env Humanoid-v2 --logging_dir $LOGDIR/td3_humanoid_2 &
pids[8]=$!

CUDA_VISIBLE_DEVICES="3" python $PKGDIR/off-policy/train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGDIR/td3_walker2d_0 &
pids[9]=$!
CUDA_VISIBLE_DEVICES="3" python $PKGDIR/off-policy/train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGDIR/td3_walker2d_1 &
pids[10]=$!
CUDA_VISIBLE_DEVICES="3" python $PKGDIR/off-policy/train.py --alg TD3 --env Walker2d-v2 --logging_dir $LOGDIR/td3_walker2d_2 &
pids[11]=$!

CUDA_VISIBLE_DEVICES="4" python $PKGDIR/off-policy/train.py --alg SAC --env Ant-v2 --logging_dir $LOGDIR/sac_ant_0 &
pids[12]=$!
CUDA_VISIBLE_DEVICES="4" python $PKGDIR/off-policy/train.py --alg SAC --env Ant-v2 --logging_dir $LOGDIR/sac_ant_1 &
pids[13]=$!
CUDA_VISIBLE_DEVICES="4" python $PKGDIR/off-policy/train.py --alg SAC --env Ant-v2 --logging_dir $LOGDIR/sac_ant_2 &
pids[14]=$!

CUDA_VISIBLE_DEVICES="5" python $PKGDIR/off-policy/train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGDIR/sac_cheetah_0 &
pids[15]=$!
CUDA_VISIBLE_DEVICES="5" python $PKGDIR/off-policy/train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGDIR/sac_cheetah_1 &
pids[16]=$!
CUDA_VISIBLE_DEVICES="5" python $PKGDIR/off-policy/train.py --alg SAC --env HalfCheetah-v2 --logging_dir $LOGDIR/sac_cheetah_2 &
pids[17]=$!

CUDA_VISIBLE_DEVICES="6" python $PKGDIR/off-policy/train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGDIR/sac_humanoid_0 &
pids[18]=$!
CUDA_VISIBLE_DEVICES="6" python $PKGDIR/off-policy/train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGDIR/sac_humanoid_1 &
pids[19]=$!
CUDA_VISIBLE_DEVICES="6" python $PKGDIR/off-policy/train.py --alg SAC --env Humanoid-v2 --logging_dir $LOGDIR/sac_humanoid_2 &
pids[20]=$!

CUDA_VISIBLE_DEVICES="7" python $PKGDIR/off-policy/train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGDIR/sac_walker2d_0 &
pids[21]=$!
CUDA_VISIBLE_DEVICES="7" python $PKGDIR/off-policy/train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGDIR/sac_walker2d_1 &
pids[22]=$!
CUDA_VISIBLE_DEVICES="7" python $PKGDIR/off-policy/train.py --alg SAC --env Walker2d-v2 --logging_dir $LOGDIR/sac_walker2d_2 &
pids[23]=$!

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
