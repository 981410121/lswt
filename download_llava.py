from huggingface_hub import hf_hub_download

# 下载单个文件
file_path = hf_hub_download(
    repo_id="liuhaotian/llava-v1.5-13b",
    filename="pytorch_model-00002-of-00003.bin",
    local_dir="./"
)
print(f"文件下载到: {file_path}")