import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================================================
# 请确保这里引用的类是你项目中实际定义的类
# =========================================================
from fastdc_qwen2_5_vl import FastDCQwen2_5_VLForConditionalGeneration

# ================= 配置 =================
MODEL_PATH = "/root/final_paper/qwen2_5_vl_3b"  # 请修改为你的模型路径
IMAGE_PATH = "/root/final_paper/qwen2_5_vl_generate/resized.jpg"  # 请修改为你的图片路径
import cv2
import numpy as np
from PIL import Image
import torch

def visualize_pruning_mask(model, processor, image_path, save_path="fastdc_high_res.png"):
    print(f"\n[Visualizing] Processing {image_path}...")
    
    # --- 1. 同样的推理逻辑 (保持不变) ---
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image."}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        _ = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw
        )

    # --- 2. 获取数据 ---
    if not hasattr(model.model, "debug_info") or "last_mask" not in model.model.debug_info:
        print("❌ Error: No debug info found.")
        return

    mask = model.model.debug_info["last_mask"]
    h_idx = model.model.debug_info["last_h"]
    w_idx = model.model.debug_info["last_w"]
    
    # 相对坐标还原
    h_min, w_min = h_idx.min(), w_idx.min()
    grid_h = int(h_idx.max() - h_min + 1)
    grid_w = int(w_idx.max() - w_min + 1)
    
    reconstructed_mask = np.zeros((grid_h, grid_w), dtype=np.float32)
    reconstructed_mask[h_idx - h_min, w_idx - w_min] = mask.astype(np.float32)

    # --- 3. 高清图像处理 (不再使用 Matplotlib) ---
    original_img = Image.open(image_path).convert("RGB")
    w, h = original_img.size
    img_arr = np.array(original_img) # [H, W, 3]

    # (A) 将 Mask 缩放到原图的像素级尺寸
    # cv2.resize 接收 (width, height)
    mask_resized = cv2.resize(reconstructed_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_expanded = mask_resized[:, :, np.newaxis] # [H, W, 1]

    # (B) 生成叠加图 (Focus View)
    alpha = 0.7 # 变暗程度
    # 逻辑：保留区域保持原样，丢弃区域变暗
    vis_img = img_arr * mask_expanded + (img_arr * (1 - alpha)) * (1 - mask_expanded)
    vis_img = vis_img.astype(np.uint8)

    # (C) 像素级拼接 (横向拼接：左原图，右结果)
    # 创建一个大画布：高度=h, 宽度=2*w
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # 填入左侧
    canvas[:, :w, :] = img_arr
    # 填入右侧
    canvas[:, w:, :] = vis_img

    # --- 4. 保存 ---
    # OpenCV 默认是 BGR 顺序，保存前需要转一下颜色空间
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)
    
    print(f"✅ High-Res Visualization saved to: {save_path}")
    print(f"   -> Resolution: {w*2}x{h} (Pixel Perfect)")
    print(f"   -> Keep Ratio: {mask.mean()*100:.2f}%")
# def visualize_pruning_mask(model, processor, image_path):
#     print("\n[Test 1] Running Visualization Test...")
    
#     # 1. 准备输入
#     # Qwen2.5-VL 建议使用 min_pixels/max_pixels 来控制分辨率，
#     # 也可以不传，让它自动处理。这里保持你的设置。
#     messages = [
#         {"role": "user", "content": [
#             {"type": "image", "image": image_path},
#             {"type": "text", "text": "图片中狗的牵引绳是什么颜色？"}
#         ]}
#     ]
    
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
    
#     inputs = processor(
#         text=[text], 
#         images=image_inputs, 
#         videos=video_inputs, 
#         padding=True, 
#         return_tensors="pt"
#     ).to(model.device)

#     # 2. 执行一次 Forward (强制触发剪枝)
#     with torch.no_grad():
#         _ = model(
#             input_ids=inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             pixel_values=inputs.pixel_values,
#             image_grid_thw=inputs.image_grid_thw
#         )

#     # 3. 获取插桩数据
#     if not hasattr(model.model, "debug_info") or "last_mask" not in model.model.debug_info:
#         print("❌ Error: Debug info not found or empty. Make sure the model ran correctly.")
#         return

#     mask = model.model.debug_info["last_mask"] # (N_visual,)
#     h_idx = model.model.debug_info["last_h"]   # (N_visual,)
#     w_idx = model.model.debug_info["last_w"]   # (N_visual,)
    
#     print(f"   -> Captured {len(mask)} visual tokens.")
#     print(f"   -> Raw H range: {h_idx.min()}~{h_idx.max()}")
#     print(f"   -> Raw W range: {w_idx.min()}~{w_idx.max()}")

#     # ================= 核心修复 =================
#     # 4. 重构 2D Mask 矩阵 (使用相对坐标)
#     # 减去最小坐标，将 Grid 移回原点 (0,0)
#     h_min = h_idx.min()
#     w_min = w_idx.min()
    
#     # 计算 Grid 的真实尺寸
#     grid_h = int(h_idx.max() - h_min + 1)
#     grid_w = int(w_idx.max() - w_min + 1)
    
#     # 创建 2D 矩阵
#     reconstructed_mask = np.zeros((grid_h, grid_w), dtype=np.float32)
    
#     # 填充 mask
#     # 注意：这里使用 h_idx - h_min 作为索引
#     # mask 中 True/1 表示保留，False/0 表示丢弃
#     reconstructed_mask[h_idx - h_min, w_idx - w_min] = mask.astype(np.float32)

#     # ================= 绘图优化 =================
#     # 5. 图像处理与叠加
#     original_img = Image.open(image_path).convert("RGB")
#     original_w, original_h = original_img.size
#     img_arr = np.array(original_img)
    
#     # 将 Mask 缩放到原图尺寸
#     # 使用 INTER_NEAREST 保持块状感 (更真实反映 Token 粒度)
#     # 如果想要平滑热力图，可以改用 INTER_LINEAR
#     mask_resized = cv2.resize(reconstructed_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
#     # 扩展 Mask 维度以匹配 RGB [H, W, 1]
#     mask_expanded = mask_resized[:, :, np.newaxis]

#     # --- 制作 "Focus View" (高亮保留区域，变暗丢弃区域) ---
#     # alpha: 丢弃区域的变暗程度 (0~1)，0.9 表示非常暗
#     alpha = 0.7 
    
#     # 逻辑：
#     # 像素值 = 原图 * Mask (保留部分保持原样) 
#     #        + 原图 * (1-Mask) * (1-alpha) (丢弃部分变暗)
#     vis_img = img_arr * mask_expanded + (img_arr * (1 - alpha)) * (1 - mask_expanded)
#     vis_img = vis_img.astype(np.uint8)

#     # --- 绘图 ---
#     plt.figure(figsize=(14, 7))
    
#     # 左图：原图
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_img)
#     plt.title(f"Original Image ({original_w}x{original_h})", fontsize=14)
#     plt.axis('off')
    
#     # 右图：剪枝结果
#     plt.subplot(1, 2, 2)
#     plt.imshow(vis_img)
    
#     keep_ratio = mask.mean() * 100
#     plt.title(f"FastDC Pruning Result\nKeep Ratio: {keep_ratio:.2f}% (Bright areas are kept)", fontsize=14)
#     plt.axis('off')
    
#     save_path = "fastdc_aligned_result_90.png"
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ Visualization saved to {save_path}")
#     print("   -> 修复验证：请检查高亮区域是否准确覆盖了物体（如杯子、苹果）。")


def run_all():
    print(f"Loading model from {MODEL_PATH}...")
    # 注意：strict=False 是为了兼容如果你在 config 里加了新参数的情况，通常可以去掉
    model = FastDCQwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 运行可视化
    visualize_pruning_mask(model, processor, IMAGE_PATH)

if __name__ == "__main__":
    run_all()