model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llm_model='BERT' # LLama7b:4096; GPT2-small:768; BERT-base:768
llm_dim=768
llm_layers=4

batch_size=48
d_model=2
n_heads=4
patch_len=4
d_ff=4
sample_rate=5 # sample every sample_rate points
seq_len=150 # 750 total
train_size=1000
valid_size=100
test_size=100

comment='TimeLLM-Zepp'

python run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Zepp/ \
  --model_id Zepp_750_1 \
  --model $model_name \
  --data Zepp \
  --prompt_domain 1 \
  --features MS \
  --seq_len $seq_len \
  --sample_rate $sample_rate \
  --label_len 1 \
  --pred_len 1 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 3 \
  --des 'Exp' \
  --itr 1 \
  --lradj 'COS'\
  --d_model $d_model \
  --patch_len $patch_len \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --train_size $train_size \
  --valid_size $valid_size \
  --test_size $test_size \
  --model_comment $comment
