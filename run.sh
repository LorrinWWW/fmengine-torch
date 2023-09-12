
##### PP
# deepspeed --num_gpus 8 --num_nodes 1 cli/llama_train.py \
#     --output_dir cache/models \
#     --init_ckpt /home/jue/v2/llama2-7b-shard \
#     --data_path /home/jue/v1/fmengine-torch/data/llama-instruct/llama-instruct-19K.jsonl \
#     --max_seq_len 4096 \
#     --train_steps 1000 \
#     --eval_steps 10 \
#     --save_steps 10 \
#     --log_steps 1 \
#     --pipe_parallel_size 2 \
#     --model_parallel_size 1 \
#     --use_flash_attn true \
#     --use_custom_llama true \
#     --deepspeed_config ./configs/llama.json

##### TP
# deepspeed --num_gpus 8 --num_nodes 1 cli/llama_train.py \
#     --output_dir cache/models \
#     --init_ckpt /home/jue/v2/llama2-7b-mp2 \
#     --data_path /home/jue/v1/fmengine-torch/data/llama-instruct/llama-instruct-19K.jsonl \
#     --max_seq_len 4096 \
#     --train_steps 1000 \
#     --eval_steps 10 \
#     --save_steps 10 \
#     --log_steps 1 \
#     --pipe_parallel_size 1 \
#     --model_parallel_size 2 \
#     --use_flash_attn true \
#     --use_custom_llama true \
#     --deepspeed_config ./configs/llama.json

deepspeed --num_gpus 8 --num_nodes 1 cli/llama_train.py \
    --output_dir /home/jue/v2/model_ckpts/lora_delay \
    --init_ckpt /home/jue/v2/codellama-34b-mp4 \
    --data_path /home/jue/v1/fmengine-torch/data/llama-instruct/llama-instruct-19K.jsonl \
    --max_seq_len 4096 \
    --train_steps 1000 \
    --eval_steps 200 \
    --save_steps 200 \
    --log_steps 1 \
    --pipe_parallel_size 1 \
    --model_parallel_size 4 \
    --use_flash_attn true \
    --use_custom_llama true \
    --deepspeed_config ./configs/llama.json


# deepspeed --num_gpus 8 --num_nodes 1 cli/llama_lora_train.py \
#     --output_dir /home/jue/v2/model_ckpts/lora_delay2 \
#     --init_ckpt /home/jue/v2/model_ckpts/lora_delay \
#     --data_path /home/jue/v1/fmengine-torch/data/llama-instruct/llama-instruct-19K.jsonl \
#     --max_seq_len 4096 \
#     --train_steps 1000 \
#     --eval_steps 200 \
#     --save_steps 200 \
#     --log_steps 1 \
#     --pipe_parallel_size 1 \
#     --model_parallel_size 4 \
#     --use_flash_attn true \
#     --use_custom_llama true \
#     --deepspeed_config ./configs/llama-lora.json

