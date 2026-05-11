#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0

model_name=DTAF
root_path=./dataset/ETT-small/

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 512 \
  --d_model 256 \
  --d_ff 256 \
  --r_dropout 0 \
  --checkpoints ./checkpoints/ETTh1_96_96 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 512 \
  --d_model 256 \
  --d_ff 256 \
  --r_dropout 0 \
  --checkpoints ./checkpoints/ETTh1_96_192 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 256 \
  --d_model 512 \
  --d_ff 512 \
  --r_dropout 0 \
  --checkpoints ./checkpoints/ETTh1_96_336 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 192 \
  --d_model 512 \
  --d_ff 512 \
  --r_dropout 0 \
  --checkpoints ./checkpoints/ETTh1_96_720 \
  --itr 1
