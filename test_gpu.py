import torch

print("PyTorch 版本:", torch.__version__)
print("PyTorch 编译的 CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
# from datasets import load_dataset

# # 将 'your_dataset_name' 替换为你正在评测的任务对应的数据集名称
# # 如果不知道名字，可以查看报错任务的 YAML 配置文件中的 dataset_path
# ds = load_dataset("/root/final_paper/datasets/Lin-Chen___mm_star")
# print(ds)