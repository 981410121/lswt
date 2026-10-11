import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
# 替换为你自己的文件名
from fastdc_qwen2_5_vl import FastDCQwen2_5_VLForConditionalGeneration

# ================= 配置 =================
MODEL_PATH = "/root/final_paper/qwen2_5_vl_3b"
IMAGE_PATH = "/root/final_paper/qwen2_5_vl_generate/resized.jpg" # 替换为你的测试图片
FASTV_LAYER_INDEX = 2

def visualize_pruning_mask(model, processor, image_path):
    """
    测试 1: 可视化测试
    核心逻辑: 利用 Qwen 的 position_ids (h, w) 将 1D Mask 还原为 2D Heatmap
    """
    print("\n[Test 1] Running Visualization Test...")
    
    # 1. 准备输入
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image."}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", max_pixels=1512*2044
    ).to(model.device)

    # 2. 执行一次 Forward (强制触发剪枝)
    with torch.no_grad():
        # 只跑 prefill 阶段即可触发剪枝
        _ = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw
        )

    # 3. 获取插桩数据
    if not hasattr(model.model, "debug_info"):
        print("❌ Error: Debug info not found. Did you add the 'self.debug_info' lines to the model code?")
        return

    mask = model.model.debug_info["last_mask"] # (N_visual,)
    h_idx = model.model.debug_info["last_h"]   # (N_visual,)
    w_idx = model.model.debug_info["last_w"]   # (N_visual,)
    
    # 4. 重构 2D Mask 矩阵
    # Qwen2.5-VL 的 position_ids 直接对应 grid 坐标，不需要复杂的 reshape
    max_h, max_w = h_idx.max() + 1, w_idx.max() + 1
    reconstructed_mask = np.zeros((max_h, max_w), dtype=np.float32)
    
    # 填充 mask: 1.0 = Keep (关注点), 0.0 = Prune (被丢弃)
    # 注意：FastDC 代码里 keep_mask=True 是保留
    for h, w, m in zip(h_idx, w_idx, mask):
        reconstructed_mask[h, w] = 1.0 if m else 0.0

    # 5. 绘图
    original_img = Image.open(image_path)
    original_w, original_h = original_img.size
    
    # 将 Mask 缩放到原图尺寸
    heatmap = cv2.resize(reconstructed_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    plt.figure(figsize=(12, 6))
    
    # 子图 1: 原图
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # 子图 2: 叠加效果 (红色区域为保留区域，变暗区域为被剪枝区域)
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    # 使用 alpha 混合显示 mask (黄色=保留，透明=剪枝)
    plt.imshow(heatmap, cmap='jet', alpha=0.5) 
    plt.title(f"Pruned Mask (Keep Ratio: {mask.mean():.2%})")
    plt.axis('off')
    
    save_path = "pruning_visualization_1.png"
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")
    print("   -> Check if the 'red/hot' areas cover the main object (e.g., the cat).")
    print("   -> If the mask looks like static noise or is shifted, the grid reconstruction logic is wrong.")

def generation_sanity_check(model, processor, image_path):
    """
    测试 2: PPL/生成一致性测试
    核心逻辑: 对比剪枝前后的输出，确保没有出现乱码
    """
    print("\n[Test 2] Running Generation Sanity Check...")
    
    prompt_text = "图片中的主要物体都有什么？苹果在图片中吗？"
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", max_pixels=1512*2044
    ).to(model.device)

    # 生成设置
    gen_kwargs = {
        "max_new_tokens": 100,
        "do_sample": False, # 使用 Greedy Search 减少随机性
        "temperature": 0.0
    }
    
    # 生成
    output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = [
        output_ids[len(inputs.input_ids[0]):] for output_ids in output_ids
    ]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("-" * 40)
    print(f"Prompt: {prompt_text}")
    print(f"Output (Pruned): {output_text}")
    print("-" * 40)
    
    # 简单的启发式检查
    if len(output_text) < 5:
        print("❌ FAILED: Output is too short (likely collapsed).")
    elif "" in output_text: # 乱码常见符号
        print("❌ FAILED: Output contains garbled characters.")
    else:
        print("✅ SUCCESS: Output looks coherent.")
        
def run_all():
    print("Loading model...")
    model = FastDCQwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 运行生成测试
    generation_sanity_check(model, processor, IMAGE_PATH)
    
    # 运行可视化测试
    visualize_pruning_mask(model, processor, IMAGE_PATH)

if __name__ == "__main__":
    run_all()