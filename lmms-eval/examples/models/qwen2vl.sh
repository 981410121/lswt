# Run and exactly reproduce qwen2vl results!
# mme as an example
# pip3 install qwen_vl_utils
export CUDA_VISIBLE_DEVICES="0"
accelerate launch \
    --num_processes=1 \
    --main_process_port=12345 \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args=pretrained=/root/final_paper/qwen2_vl_2b,max_pixels=1179648 \
    --tasks mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path ./logs/