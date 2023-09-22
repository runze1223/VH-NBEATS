if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi


seq_len=336
model_name=VH-NBEATS
embedding=3

for pred_len in 96
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  traffic_basis\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862  \
    --dec_in 862  \
    --c_out 862 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --duplicate 1\
    --decomposition 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --embedding $embedding \
    --itr 1 --batch_size 6 --learning_rate 0.0001  >logs/LongForecasting_new/Traffic_3_$model_name'_'$seq_len'_'$pred_len.log 
done


embedding=1

for pred_len in 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  traffic_basis\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862  \
    --dec_in 862  \
    --c_out 862 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --duplicate 1\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --embedding $embedding \
    --itr 1 --batch_size 6 --learning_rate 0.0001  >logs/LongForecasting_new/Traffic_1_two_$model_name'_'$seq_len'_'$pred_len.log
done


seq_len=336
model_name=VH-PatchTST
embedding=3

for pred_len in 96
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  traffic_basis\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862  \
    --dec_in 862  \
    --c_out 862 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --embedding $embedding \
    --itr 1 --batch_size 6 --learning_rate 0.0001  >logs/LongForecasting_new/Traffic_3_$model_name'_'$seq_len'_'$pred_len.log 
done



embedding=1

for pred_len in 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  traffic_basis\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862  \
    --dec_in 862  \
    --c_out 862 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --embedding $embedding \
    --itr 1 --batch_size 6 --learning_rate 0.0001  >logs/LongForecasting_new/Traffic_1_$model_name'_'$seq_len'_'$pred_len.log 
done


seq_len=336
model_name=PatchTST

for pred_len in 96 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id  traffic_basis\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 862  \
    --dec_in 862  \
    --c_out 862 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --decomposition 0\
    --stride 8\
    --des 'Exp' \
    --train_epochs 50\
    --itr 1 --batch_size 6 --learning_rate 0.0001  >logs/LongForecasting_new/Traffic_$model_name'_'$seq_len'_'$pred_len.log 
done





