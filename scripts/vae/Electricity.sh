if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi


seq_len=336
model_name=VH-NBEATS
embedding=1 

for pred_len in 96 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  electricity_basis1\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --variation 1\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --embedding $embedding\
    --beta 1\
    --duplicate 1 \
    --itr 1 --batch_size 16 --learning_rate 0.0001  >logs/LongForecasting_new/electricity_var1_1_two_$model_name'_'$seq_len'_'$pred_len.log 
done

seq_len=336
model_name=VH-PatchTST
embedding=1 

for pred_len in 96 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  electricity_basis1\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
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
    --train_epochs 100\
    --embedding $embedding\
    --itr 1 --batch_size 16 --learning_rate 0.0001  >logs/LongForecasting_new/electricity_1_$model_name'_'$seq_len'_'$pred_len.log 
done

seq_len=336
model_name=PatchTST
embedding=1 

for pred_len in 96 192 336 720 
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id  electricity_basis1\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len\
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321\
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
    --train_epochs 100\
    --embedding $embedding\
    --itr 1 --batch_size 16 --learning_rate 0.0001  >logs/LongForecasting_new/electricity_$model_name'_'$seq_len'_'$pred_len.log 
done






