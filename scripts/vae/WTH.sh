if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi


seq_len=336
model_name=VH-NBEATS
embedding=0 
embedding2=3 

for pred_len in 96 192 336 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path WTH.csv \
    --model_id  WTH_basis03\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12\
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.35\
    --fc_dropout 0.35\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --individual_embed 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --embedding $embedding2\
    --beta 1\
    --itr 1 --batch_size 64 --learning_rate 0.0001  >logs/LongForecasting_new/Weather_3_$model_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path WTH.csv \
    --model_id  WTH_basis03\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12\
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.35\
    --fc_dropout 0.35\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --individual_embed 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --embedding $embedding $embedding2\
    --beta 1\
    --itr 1 --batch_size 64 --learning_rate 0.0001  >logs/LongForecasting_new/Weather_03_$model_name'_'$seq_len'_'$pred_len.log 
done

seq_len=336
model_name=VH-PatchTST
embedding=0 
embedding2=3 

for pred_len in 96 192 336
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path WTH.csv \
    --model_id  WTH_basis03\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12\
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.35\
    --fc_dropout 0.35\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --individual_embed 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --embedding  $embedding2\
    --beta 1\
    --itr 1 --batch_size 64 --learning_rate 0.0001  >logs/LongForecasting_new/Weather_3_$model_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path WTH.csv \
    --model_id  WTH_basis03\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12\
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.35\
    --fc_dropout 0.35\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --individual_embed 0\
    --stride 8\
    --des 'Exp'\
    --train_epochs 100\
    --embedding $embedding $embedding2\
    --beta 1\
    --itr 1 --batch_size 64 --learning_rate 0.0001  >logs/LongForecasting_new/Weather_03_$model_name'_'$seq_len'_'$pred_len.log 
done

seq_len=336
model_name=PatchTST
embedding=0 
embedding2=3 

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path WTH.csv \
    --model_id  WTH_basis03\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12\
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.35\
    --fc_dropout 0.35\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --individual_embed 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --embedding $embedding $embedding2\
    --itr 1 --batch_size 64 --learning_rate 0.0001  >logs/LongForecasting_new/Weather_03_$model_name'_'$seq_len'_'$pred_len.log 
done
