#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0

model_name=DTAF
root_path=./dataset/ETT-small/

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --split ett \
  --drop_last \
  --seq_len 512 \
  --pred_len 96 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.005 \
  --lradj type1 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.1 \
  --k 1 \
  --r_dropout 0.001 \
  --patch_len 8 \
  --aggregated_norm 0 \
  --heads 1 \
  --sigma 1.0 \
  --expert_num 2 \
  --kan_div 8 \
  --kl 0.1 \
  --checkpoints ./checkpoints/ETTh2_96_96 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --split ett \
  --drop_last \
  --seq_len 512 \
  --pred_len 192 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.005 \
  --lradj type1 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.1 \
  --k 1 \
  --r_dropout 0.0001 \
  --patch_len 16 \
  --aggregated_norm 0 \
  --heads 1 \
  --sigma 2.0 \
  --expert_num 2 \
  --kan_div 8 \
  --kl 0.5 \
  --checkpoints ./checkpoints/ETTh2_96_192 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --split ett \
  --drop_last \
  --seq_len 512 \
  --pred_len 336 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.005 \
  --lradj type1 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.1 \
  --k 3 \
  --r_dropout 0.0001 \
  --patch_len 16 \
  --aggregated_norm 1 \
  --heads 4 \
  --sigma 1.0 \
  --expert_num 2 \
  --kan_div 4 \
  --kl 0.05 \
  --checkpoints ./checkpoints/ETTh2_96_336 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --split ett \
  --drop_last \
  --seq_len 512 \
  --pred_len 720 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.005 \
  --lradj type1 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.1 \
  --k 1 \
  --r_dropout 0.0001 \
  --patch_len 48 \
  --aggregated_norm 0 \
  --heads 4 \
  --sigma 0.5 \
  --expert_num 2 \
  --kan_div 1 \
  --kl 0.1 \
  --checkpoints ./checkpoints/ETTh2_96_720 \
  --itr 1
