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
export CUDA_VISIBLE_DEVICES="2,3"
accelerate launch \
    --num_processes=2 \
    --main_process_port=12345 \
    -m lmms_eval \
    --model llava \
    --model_args "pretrained=/root/data1/final_paper/liuhaotian_llava_7b,device_map=auto" \
    --tasks flickr30k_test_lite,scienceqa_img,vizwiz_vqa_val,mmmu_val,pope,ok_vqa,mme,mmstar,mmbench_en_dev,nocaps_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path ./ablation_study/llava_output_fastdc_wo_window_2_50_10bench_1224_logs/