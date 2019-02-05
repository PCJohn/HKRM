#!/usr/bin/env bash
#SBATCH --job-name=spat_ade
#SBATCH -o gypsum/logs/%j_spat_ade.txt 
#SBATCH -e gypsum/errs/%j_spat_ade.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



# Recreate Spatial training on ADE20k


DATASET=ade
# set net to Attribute, Relation, Spatial or HKRM
NET=Relation
EXP_NAME=$DATASET"_"$NET

mkdir Outputs/$EXP_NAME
mkdir Outputs/$EXP_NAME/logs

python trainval_HKRM.py \
    --dataset $DATASET \
    --bs 2 --nw 4 \
    --log_dir Outputs/$EXP_NAME/logs \
    --save_dir Outputs/$EXP_NAME \
    --init --net $NET \
    --attr_size 256 --rela_size 256 --spat_size 256 \


