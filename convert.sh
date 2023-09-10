rm -rf  ~/v2/codellama-34b-mp4
python scripts/conversions/llama/from_hf.py \
    --model_name_or_path ~/v2/CodeLlama-34b \
    --output_dir ~/v2/codellama-34b-mp4 --mp_world_size 4