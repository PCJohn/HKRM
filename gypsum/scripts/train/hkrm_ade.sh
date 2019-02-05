#!/usr/bin/env bash
#SBATCH --job-name=hkrm_ade
#SBATCH -o gypsum/logs/%j_hkrm_ade.txt 
#SBATCH -e gypsum/errs/%j_hkrm_ade.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



# starts from WIDER pre-trained model
# trial run: just using CS6 data (imdb merging not done)


CUDA_VISIBLE_DEVICES=$GPU_ID \
python trainval_HKRM.py \
    --dataset ade \
    --bs 2 \
    --nw 4 \
    --log_dir $LOG_DIR \
    --save_dir $WHERE_YOU_WANT \
    --init --net HKRM --attr_size 256 --rela_size 256 --spat_size 256 \

