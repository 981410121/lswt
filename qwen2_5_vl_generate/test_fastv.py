import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
# 确保这里导入的是你修改后的类
from fastv_qwen2_5_vl import FastVQwen2_5_VLForConditionalGeneration
import numpy as np

# ================= 配置 =================
MODEL_PATH = "/root/final_paper/qwen2_5_vl_3b" 
FASTV_LAYER_INDEX = 2  # 必须与模型内部硬编码的 FASTV_k 一致

# --- [新增] 显存监控辅助函数 ---
def print_peak_memory(tag: str):
    if torch.cuda.is_available():
        # 获取当前记录周期内的最大分配显存 (Tensor占用)
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # 获取当前记录周期内的最大保留显存 (Cache + Tensor, 显存条上实际占用的)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"[{tag}] Peak Allocated: {peak_allocated:.2f} GB | Peak Reserved: {peak_reserved:.2f} GB")
        
        # 重置统计，以便下一阶段重新计算峰值
        torch.cuda.reset_peak_memory_stats()
    else:
        print(f"[{tag}] CUDA not available.")

def run_test():
    # 0. 初始化显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = FastVQwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # --- [新增] 打印模型权重占用的显存 ---
    print_peak_memory("Model Weights Load")

    # ================= 注册 Hook (保持原样) =================
    stats_log = []

    def before_layer_hook(module, args, kwargs):
        hidden_states = kwargs.get('hidden_states')
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]
            
        position_ids = kwargs.get('position_ids')
        if position_ids is None and len(args) > 2:
            position_ids = args[2]

        if hidden_states is not None:
            seq_len = hidden_states.shape[1]
            if seq_len > 1:
                stats_log.append({
                    "stage": "pre_hook",
                    "len": seq_len,
                    "pos_ids_shape": position_ids.shape if position_ids is not None else "Unknown"
                })

    def after_layer_hook(module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        seq_len = hidden_states.shape[1]
        
        if len(stats_log) > 0 and stats_log[-1]["stage"] == "pre_hook":
            stats_log[-1]["pruned_len"] = seq_len
            stats_log[-1]["stage"] = "completed"

    target_layer = model.model.layers[FASTV_LAYER_INDEX]
    try:
        target_layer.register_forward_pre_hook(before_layer_hook, with_kwargs=True)
    except TypeError:
        target_layer.register_forward_pre_hook(lambda m, args: before_layer_hook(m, args, {}))
        
    target_layer.register_forward_hook(after_layer_hook)
    print(f"Hook registered on Layer {FASTV_LAYER_INDEX}.")

    # 2. 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/root/final_paper/qwen2_5_vl_generate/be13136077895.jpg",
                },
                {"type": "text", "text": "详细描述这张图片的内容。"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    MAX_PIXELS = 840 * 840
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=MAX_PIXELS
    )
    inputs = inputs.to(model.device)

    # 3. 推理
    print("\nStarting Generation...")
    
    # --- [新增] 在推理前重置统计，专门捕获推理过程中的峰值 ---
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    # --- [新增] 打印推理过程的显存峰值 ---
    print_peak_memory("Inference (Prefill + Decode)")

    # 4. 打印结果
    print("\n" + "="*20 + " Pruning Analysis " + "="*20)
    
    prefill_stats = [s for s in stats_log if "pruned_len" in s]
    
    if prefill_stats:
        stats = prefill_stats[0]
        vis_before = stats['len']
        vis_after = stats['pruned_len']
        
        print(f"Layer Index: {FASTV_LAYER_INDEX}")
        print(f"[Input]  Sequence Length: {vis_before}")
        print(f"[Output] Sequence Length: {vis_after}")
        print(f"[Diff]   Pruned Tokens:   {vis_before - vis_after}")
        print(f"[Rate]   Retention Rate:  {vis_after / vis_before:.2%}")
        
        print(f"\n[Position IDs Check]")
        print(f"Original Pos IDs Shape: {stats['pos_ids_shape']}")
        
        if vis_before > vis_after:
            print("\n✅ SUCCESS: Pruning logic triggered successfully!")
        else:
            print("\n❌ WARNING: No tokens pruned. Check if layer index matches logic.")
    else:
        print("\n❌ ERROR: No prefill stage detected.")

if __name__ == "__main__":
    run_test()