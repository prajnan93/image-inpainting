#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=inpaint_exp1
#SBATCH --partition=jiang
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=../../outs/v11/exp_1.out


module load cuda/11.3
cd ..
python train.py --train_ds_dir './samples/Places365' \
                --val_ds_dir './samples/Places365' \
                --CKPT_DIR '../../experiments/inpaint/ckpts/exp1' \
                --LOG_DIR '../../experiments/inpaint/logs/exp1' \
                --SAMPLE_DIR '../../experiments/inpaint/samples/exp1' \
                --crop_size 384 384 \
                --mask_type 'free_form' \
                --mask_num 20 \
                --max_angle 10 \
                --max_len 40 \
                --max_width 50 \
                --margin 10 10 \
                --bbox_shape 100 100 \
                --epochs 50 \
                --batch_size 5 \
                --num_workers 1 \
                --device_id 0 \
                --lr_g 1e-4 \
                --lr_d 1e-4 \
                --b1 0.5 \
                --b2 0.999 \
                --weight_decay 0 \
                --lambda_l1 10 \
                --lambda_perceptual 10 \
                --lambda_gan 1 \
                --lr_decrease_epoch 10 \
                --lr_decrease_factor 0.5 \
                --LOG_INTERVAL 100 \
                --SAVE_SAMPLES_INTERVAL 5 \
                --SAVE_SAMPLE_COUNT 10 \
                --use_cuda \
                --in_channels 4 \
                --out_channels 3 \
                --latent_channels 64 \
                --pad_type 'zero' \
                --activation 'elu' \
                --norm_d 'none' \
                --norm_g 'none' \
                --init_type 'kaiming' \
                --init_gain 0.02 \
                --use_perceptualnet
