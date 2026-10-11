# # install lmms_eval without building dependencies
# cd lmms_eval;
# pip install --no-deps -U -e .

# # install LLaVA without building dependencies
# cd LLaVA
# pip install --no-deps -U -e .

# # install all the requirements that require for reproduce llava results
# pip install -r llava_repr_requirements.txt

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
# accelerate launch --num_processes=2 -m lmms_eval --model llava   --model_args pretrained="/root/data1/final_paper/liuhaotian_llava_7b,device_map=auto"   --tasks mmbench_en  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/
export CUDA_VISIBLE_DEVICES="1"
accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args "pretrained=/root/data1/final_paper/liuhaotian_llava_7b,device_map=auto" \
    --tasks scienceqa_img,nocaps_val,flickr30k_test \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path ./llava_output_fastdc_w_dc_window_dc_3bench_4_4_1221_logs/