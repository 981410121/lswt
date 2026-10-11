## 离线数据备份

由于测评数据文件较大，没有提交到 GitHub。当前本地已有的 MME 数据已备份到百度网盘：

- 分享链接：[https://pan.baidu.com/s/1QpvsiI-c03rPaSvrsL0b9w?pwd=7011](https://pan.baidu.com/s/1QpvsiI-c03rPaSvrsL0b9w?pwd=7011)
- 提取码：`7011`
- 网盘内容：`no/datasets/lmms-lab___mme/`

模型权重未包含在 GitHub 或该网盘备份中，使用时需另行下载 Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B/13B 等预训练权重。其余评测数据集也需通过 Hugging Face、LMMS-Eval 或数据集官方渠道获取。

## 说明

部分脚本保留了原实验环境中的绝对路径（如 `/root/final_paper/...`），在新机器运行前需要修改为实际模型和数据路径。
