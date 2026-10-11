import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区 =================
# 1. 模型路径
MODEL_PATH = "/root/final_paper/qwen2_5_vl_3b" 

# 2. 图片地址 (可以是 URL 或 本地路径)
IMAGE_PATH = "/root/final_paper/qwen2_5_vl_generate/be13136077895.jpg"
# IMAGE_PATH = "/root/my_image.jpg" # 本地路径示例

# 3. 你想要测试的 max_pixels 设置
# 尝试修改这个值，比如 512*512, 1280*1280, 或者 None (默认)
TEST_MAX_PIXELS = 512 * 512  
# TEST_MAX_PIXELS = None # 使用默认策略

def check_image_size():
    print(f"{'='*20} 1. 原始图片检查 {'='*20}")
    
    # --- 加载图片 ---
    if IMAGE_PATH.startswith("http"):
        response = requests.get(IMAGE_PATH)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(IMAGE_PATH)
    
    w, h = image.size
    print(f"图片来源: {IMAGE_PATH}")
    print(f"原始宽 (Width):  {w}")
    print(f"原始高 (Height): {h}")
    print(f"原始像素总数:    {w * h} (约 {(w*h)/10000:.1f} 万像素)")
    
    print(f"\n{'='*20} 2. Qwen Processor 处理后检查 {'='*20}")
    print(f"当前设置的 max_pixels: {TEST_MAX_PIXELS}")

    # --- 加载 Processor ---
    # min_pixels 默认通常是 256*256
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # --- 构造输入 ---
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "Describe this."},
        ]}
    ]
    
    # --- 预处理 ---
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 这里我们传入 max_pixels 参数进行模拟
    inputs = processor(
        text=["Describe this."],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=TEST_MAX_PIXELS, # <--- 关键参数
    )
    
    # --- 分析结果 ---
    # Qwen2.5-VL 的 inputs 里有一个 'image_grid_thw' 
    # 格式通常是 [Batch, 3] -> [T, H_grid, W_grid]
    # H_grid 和 W_grid 是 Patch 的数量 (每个 Patch 是 14x14)
    
    grid_thw = inputs['image_grid_thw']
    
    # 假设只有一张图，取第一行
    t, h_grid, w_grid = grid_thw[0].tolist()
    
    # 计算实际分辨率
    processed_h = h_grid * 14
    processed_w = w_grid * 14
    total_tokens = h_grid * w_grid
    
    print(f"模型实际处理的 Patch 格子 (Grid): {h_grid} x {w_grid}")
    print(f"模型实际看到的图片分辨率: {processed_w} x {processed_h}")
    print(f"实际像素总数: {processed_w * processed_h}")
    print(f"视觉 Token 数量 (约等于显存占用): {total_tokens}")
    
    # --- 显存估算预警 ---
    if total_tokens > 4000:
        print("\n⚠️  警告: Token 数量极大 (>4000)，显存极易 OOM！")
    elif total_tokens < 1000:
        print("\n✅ 安全: Token 数量较少，显存占用低。")
    else:
        print("\nℹ️  正常: Token 数量适中。")

if __name__ == "__main__":
    check_image_size()