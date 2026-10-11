# import torch
# import time
# import os
# import copy
# import numpy as np
# from PIL import Image
# from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
#     Qwen2_5_VLModel,
#     Qwen2_5_VLForConditionalGeneration as OriginalForConditionalGeneration,
#     BaseModelOutputWithPast
# )
# from qwen_vl_utils import process_vision_info

# # === 修复日志警告：禁用 Tokenizers 并行以避免死锁警告 ===
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # ==============================================================================
# # 1. 你的 FastDC 算法类定义 
# # ==============================================================================

# class FastDCQwen2_5_VLModel(Qwen2_5_VLModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.last_attention = None 
#         self.temporary_input_ids = None 

#     def _prune_visual_tokens(self, hidden_states, position_ids, attention_mask, input_ids, layer_config):
#         import math
        
#         # --- 安全检查 ---
#         if input_ids is None: return hidden_states, position_ids, attention_mask

#         FASTV_r = layer_config.get("FASTV_r", 0.5)
#         rank_fusion_weight = float(layer_config.get("rank_fusion_weight", 0.5))
#         WIN_H = layer_config.get("WIN_H", 8) 
#         WIN_W = layer_config.get("WIN_W", 8)
        
#         bsz, seq_len, _ = hidden_states.shape
#         device = hidden_states.device
        
#         # 仅支持 batch_size = 1
#         if bsz > 1: return hidden_states, position_ids, attention_mask
            
#         # 1. 识别视觉 Token
#         image_token_id = self.config.image_token_id
#         if input_ids.shape[0] != bsz: return hidden_states, position_ids, attention_mask

#         is_visual = (input_ids[0] == image_token_id)
#         if not is_visual.any(): return hidden_states, position_ids, attention_mask

#         visual_indices = torch.where(is_visual)[0]
        
#         # 2. 获取分数
#         if self.last_attention is None: return hidden_states, position_ids, attention_mask

#         # 注意：Qwen2.5-VL 的 Attention 输出
#         attn_weights = torch.mean(self.last_attention, dim=1) 
#         attn_img = attn_weights[0, :, is_visual] 
        
#         n_seq, n_img = attn_img.shape
#         if n_img == 0: return hidden_states, position_ids, attention_mask

#         a_lp = 1.0 / n_img
#         a_hp = attn_img - a_lp
#         score_freq_raw = torch.sum(a_hp, dim=0) / n_seq 
#         score_fastv_raw = attn_img[-1, :] 

#         # 3. 动态恢复 Grid
#         vis_pos_h = position_ids[1, 0, is_visual] 
#         vis_pos_w = position_ids[2, 0, is_visual] 

#         unique_h = torch.unique(vis_pos_h, sorted=True)
#         unique_w = torch.unique(vis_pos_w, sorted=True)

#         grid_h_indices = torch.searchsorted(unique_h, vis_pos_h)
#         grid_w_indices = torch.searchsorted(unique_w, vis_pos_w)

#         GRID_H_SIZE = len(unique_h)
#         GRID_W_SIZE = len(unique_w)

#         num_win_h = math.ceil(GRID_H_SIZE / WIN_H)
#         num_win_w = math.ceil(GRID_W_SIZE / WIN_W)

#         keep_mask_visual = torch.zeros_like(vis_pos_h, dtype=torch.bool, device=device)

#         # 4. 窗口遍历与剪枝
#         for gh in range(num_win_h):
#             for gw in range(num_win_w):
#                 h_start = gh * WIN_H
#                 h_end = min((gh + 1) * WIN_H, GRID_H_SIZE)
#                 w_start = gw * WIN_W
#                 w_end = min((gw + 1) * WIN_W, GRID_W_SIZE)
                
#                 in_window_mask = (
#                     (grid_h_indices >= h_start) & (grid_h_indices < h_end) &
#                     (grid_w_indices >= w_start) & (grid_w_indices < w_end)
#                 )
                
#                 window_indices = torch.where(in_window_mask)[0]
#                 if len(window_indices) == 0: continue

#                 local_freq_raw = score_freq_raw[window_indices] 
#                 local_fastv_raw = score_fastv_raw[window_indices]
                
#                 _, local_freq_sorted_idx = torch.sort(local_freq_raw, descending=True)
#                 _, local_fastv_sorted_idx = torch.sort(local_fastv_raw, descending=True)
                
#                 num_tokens_in_window = len(window_indices)
#                 local_weighting = torch.arange(num_tokens_in_window, 0, -1, device=device, dtype=torch.float32)
                
#                 local_rank_score = torch.zeros(num_tokens_in_window, device=device)
#                 local_rank_score[local_freq_sorted_idx] += local_weighting * rank_fusion_weight
#                 local_rank_score[local_fastv_sorted_idx] += local_weighting * (1 - rank_fusion_weight)
                
#                 local_keep_k = int(num_tokens_in_window * (1 - FASTV_r))
                
#                 if local_keep_k > 0:
#                     local_top_k_indices_in_window = local_rank_score.topk(local_keep_k).indices
#                     global_indices_to_keep = window_indices[local_top_k_indices_in_window]
#                     keep_mask_visual[global_indices_to_keep] = True

#         # 5. 应用剪枝
#         global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
#         global_keep_mask[is_visual] = keep_mask_visual
        
#         # Debug info
#         if not hasattr(self, "debug_info"): self.debug_info = {}
#         self.debug_info["pruned_count"] = (~keep_mask_visual).sum().item()
#         self.debug_info["total_visual"] = len(keep_mask_visual)

#         pruned_hidden_states = hidden_states[:, global_keep_mask, :]
        
#         # position_ids: [3, bsz, seq] -> slice dim 2
#         pruned_position_ids = position_ids[:, :, global_keep_mask]
        
#         if attention_mask is not None:
#             pruned_attention_mask = attention_mask[:, global_keep_mask]
#         else:
#             pruned_attention_mask = None

#         return pruned_hidden_states, pruned_position_ids, pruned_attention_mask

#     def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
#                 inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
#                 return_dict=None, cache_position=None):
        
#         # 找回 input_ids
#         if input_ids is None and hasattr(self, "temporary_input_ids"):
#             input_ids = self.temporary_input_ids

#         # 配置参数 (你可以根据需要调整)
#         FASTV_k = 2 
#         layer_config = {
#             "FASTV_r": 0.5, # 剪枝比例
#             "rank_fusion_weight": 0.5,
#             "WIN_H": 4, 
#             "WIN_W": 4 
#         }

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )

#         if position_ids is None:
#             position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#         )

#         hidden_states = inputs_embeds
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             layer_idx = decoder_layer.self_attn.layer_idx

#             temp_output_attentions = output_attentions
#             if layer_idx == FASTV_k - 1:
#                 temp_output_attentions = True

#             # === FastDC 剪枝逻辑 ===
#             if layer_idx == FASTV_k and self.last_attention is not None:
#                 # 仅在 prefill 阶段 (seq_len > 1) 且有 visual tokens 时尝试剪枝
#                 if hidden_states.shape[1] > 1:
#                     hidden_states, position_ids, attention_mask = self._prune_visual_tokens(
#                         hidden_states, position_ids, attention_mask, input_ids, layer_config
#                     )
                    
#                     # 剪枝后必须重新计算 causal mask 和 rotary embeddings
#                     if cache_position is not None:
#                         # 截断 cache_position 以匹配新的长度
#                         cache_position = cache_position[:hidden_states.shape[1]]
                    
#                     causal_mask = self._update_causal_mask(
#                          attention_mask, hidden_states, cache_position, past_key_values, output_attentions
#                     )
#                     position_embeddings = self.rotary_emb(hidden_states, position_ids)

#             # Gradient Checkpointing logic omitted for brevity in testing
#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=causal_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_values,
#                 output_attentions=temp_output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#             )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if temp_output_attentions else 1]

#             if layer_idx == FASTV_k - 1:
#                 self.last_attention = layer_outputs[1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None

#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

# class FastDCQwen2_5_VLForConditionalGeneration(OriginalForConditionalGeneration):
#     def __init__(self, config):
#         super().__init__(config)
#         # 替换内部 model
#         self.model = FastDCQwen2_5_VLModel(config)
#         self.post_init()
        
#     def forward(self, input_ids=None, **kwargs):
#         if hasattr(self, "model") and input_ids is not None:
#             self.model.temporary_input_ids = input_ids
#         return super().forward(input_ids=input_ids, **kwargs)

# # ==============================================================================
# # 2. 工具函数：构造 Qwen2.5-VL 的真实输入
# # ==============================================================================

# def get_test_inputs(processor, device, height=1024, width=1024):
#     """
#     构造一个包含随机噪声图片的输入，用于触发模型的视觉处理路径。
#     """
#     # 1. 创建随机图像
#     image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    
#     # 2. 构造 Prompt
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": "Describe this image in detail."},
#             ],
#         }
#     ]
    
#     # 3. 处理输入
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
    
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
    
#     return inputs.to(device)

# def get_cache_size(past_key_values):
#     """计算 DynamicCache 的显存占用 (GB)"""
#     if past_key_values is None: return 0
#     total_bytes = 0
#     # DynamicCache 结构
#     if hasattr(past_key_values, "key_cache"):
#         for k in past_key_values.key_cache:
#             total_bytes += k.element_size() * k.nelement()
#         for v in past_key_values.value_cache:
#             total_bytes += v.element_size() * v.nelement()
#     return total_bytes / (1024**3)

# # ==============================================================================
# # 3. 性能测试主流程
# # ==============================================================================

# def run_benchmark(model, inputs, max_new_tokens=50, name="Model"):
#     print(f"\n>>> Benchmarking: {name}")
#     device = model.device
#     model.eval()
    
#     # === Warmup ===
#     # 简单的 warmup，让 CUDA 初始化
#     print("  Warmup...")
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=2, use_cache=True)
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
    
#     # === 1. Prefilling Test (Time & Peak Memory) ===
#     print("  Testing Prefilling...")
#     torch.cuda.reset_peak_memory_stats()
#     start_time = time.time()
    
#     with torch.no_grad():
#         # 手动执行一次 forward 来模拟 prefill
#         outputs = model(**inputs, use_cache=True)
    
#     torch.cuda.synchronize()
#     prefill_time = time.time() - start_time
#     peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
#     # 计算 Cache Size
#     cache_size = get_cache_size(outputs.past_key_values)
    
#     # 检查是否发生剪枝 (Debug info)
#     if hasattr(model.model, "debug_info") and "pruned_count" in model.model.debug_info:
#         info = model.model.debug_info
#         print(f"  [Info] Visual Tokens: {info['total_visual']} -> Pruned: {info['pruned_count']} "
#               f"({info['pruned_count']/info['total_visual']:.1%})")

#     # === 2. Decoding Latency Test ===
#     print("  Testing Decoding...")
#     # 重新生成，这次只测 generate 循环
#     start_time = time.time()
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
#     torch.cuda.synchronize()
#     total_gen_time = time.time() - start_time
    
#     # 粗略估算 decoding latency (减去 prefill 时间)
#     decoding_time = total_gen_time - prefill_time
#     if decoding_time < 0: decoding_time = 0 
#     avg_latency = (decoding_time / max_new_tokens) * 1000 # ms/token
    
#     results = {
#         "Peak Memory (GB)": peak_memory,
#         "Prefill Time (s)": prefill_time,
#         "Decoding Latency (ms)": avg_latency,
#         "Cache Size (GB)": cache_size
#     }
#     print(f"  Results: {results}")
#     return results

# def main():
#     # 设置环境
#     model_path = "/root/final_paper/qwen2_5_vl_3b"  # 你的本地路径
#     device = "cuda:0"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
#     print(f"Loading processor from {model_path}...")
#     try:
#         processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1024*28*28)
#     except Exception as e:
#         print(f"Error loading processor: {e}")
#         return

#     # 1. 构造输入 (840x840 图像)
#     print("Generating test inputs...")
#     inputs = get_test_inputs(processor, device, height=840, width=840)
#     print(f"Input shape (Input IDs): {inputs.input_ids.shape}")

#     # 2. 加载原版模型
#     print("Loading Original Qwen2.5-VL (Eager Mode)...")
#     model = OriginalForConditionalGeneration.from_pretrained(
#         model_path, 
#         torch_dtype=torch.bfloat16, 
#         attn_implementation="eager",  # <--- 【修改点】改为 eager
#         device_map=device
#     )
    
#     # 运行原版测试
#     stats_orig = run_benchmark(model, inputs, max_new_tokens=50, name="Original")
    
#     # 3. 替换为 FastDC 模型
#     print("\nSwapping to FastDC Algorithm...")
#     config = model.config # 这里获取的 config 已经包含了 attn_implementation="eager"
    
#     # 初始化你的加速模型
#     fast_model = FastDCQwen2_5_VLForConditionalGeneration(config)
    
#     # 关键：加载原版权重
#     print("  Transferring weights...")
#     fast_model.load_state_dict(model.state_dict(), strict=False)
#     fast_model.to(dtype=torch.bfloat16, device=device)
    
#     # 释放原版模型显存
#     del model
#     torch.cuda.empty_cache()
    
#     # 运行 FastDC 测试
#     stats_fast = run_benchmark(fast_model, inputs, max_new_tokens=50, name="FastDC")
    
#     # 4. 结果对比
#     print("\n" + "="*60)
#     print(f"{'Metric':<25} | {'Original':<12} | {'FastDC':<12} | {'Diff':<10}")
#     print("-" * 60)
#     for k in stats_orig.keys():
#         v1 = stats_orig[k]
#         v2 = stats_fast[k]
#         diff = (v2 - v1) / v1 * 100
#         print(f"{k:<25} | {v1:<12.4f} | {v2:<12.4f} | {diff:+.2f}%")
#     print("="*60)

# if __name__ == "__main__":
#     main()
import torch
import time
import os
import gc  # 引入垃圾回收模块
import copy
import numpy as np
from PIL import Image
from transformers.cache_utils import DynamicCache
from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration as OriginalForConditionalGeneration,
    BaseModelOutputWithPast
)
from qwen_vl_utils import process_vision_info

# === 禁用 Tokenizers 并行以避免警告 ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# 1. FastDC 类定义 (保持不变)
# ==============================================================================

class FastDCQwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.last_attention = None 
        self.temporary_input_ids = None 

    def _prune_visual_tokens(
        self, 
        hidden_states: torch.Tensor, 
        position_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, 
        input_ids: torch.LongTensor,
        layer_config: dict
    ):
        """
        执行 FastV 剪枝逻辑 (适配 Qwen2.5-VL)
        逻辑：依据 Last Token 对视觉 Token 的 Attention Score 进行全局排序，保留 Top-K
        """
        # --- 1. 基础检查 ---
        if input_ids is None:
            return hidden_states, position_ids, attention_mask

        # FastV 只主要关注保留比例，这里复用 FASTV_r 作为"剪枝比例"
        # 如果 FASTV_r = 0.5，则保留 (1-0.5) = 50% 的 token
        prune_ratio = layer_config.get("FASTV_r", 0.5)
        
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 目前主要适配 BS=1 的推理场景
        if bsz > 1:
            return hidden_states, position_ids, attention_mask
            
        # --- 2. 识别视觉 Token ---
        image_token_id = self.config.image_token_id
        
        if input_ids.shape[0] != bsz:
             return hidden_states, position_ids, attention_mask

        is_visual = (input_ids[0] == image_token_id) # [Seq_Len]
        
        # 如果没有视觉 token，直接返回
        if not is_visual.any():
            return hidden_states, position_ids, attention_mask

        visual_indices_global = torch.where(is_visual)[0]
        num_visual_tokens = len(visual_indices_global)

        # --- 3. 计算 FastV 分数 ---
        if self.last_attention is None:
             return hidden_states, position_ids, attention_mask

        # self.last_attention shape: [BS, Num_Heads, Seq_Len, Seq_Len]
        # 对多头取平均
        attn_weights = torch.mean(self.last_attention, dim=1) # [BS, Seq_Len, Seq_Len]
        
        # FastV 核心：取最后一个 Token (text query 的末尾) 对所有 Visual Tokens 的关注度
        # attn_weights[0, -1, :] 代表序列中最后一个 token 对所有位置的注意力
        # 我们只取 visual 部分
        fastv_scores = attn_weights[0, -1, is_visual] # [Num_Visual]
        
        # --- 4. 计算保留数量与 Top-K 选择 ---
        num_to_keep = int(num_visual_tokens * (1 - prune_ratio))
        
        # 如果保留数量 <= 0，或者不需要剪枝，做相应处理
        if num_to_keep <= 0:
            # 极端情况：全部剪掉，只保留文本？通常至少保留一点，这里简单处理为全部剪掉
            keep_mask_visual = torch.zeros_like(fastv_scores, dtype=torch.bool)
        elif num_to_keep >= num_visual_tokens:
            # 不剪枝
            keep_mask_visual = torch.ones_like(fastv_scores, dtype=torch.bool)
        else:
            # 获取分数最高的 K 个索引 (Local Indices within visual tokens)
            topk_values, topk_indices_local = fastv_scores.topk(num_to_keep)
            
            keep_mask_visual = torch.zeros_like(fastv_scores, dtype=torch.bool)
            keep_mask_visual[topk_indices_local] = True

        # --- 5. 构建全局 Mask ---
        global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        
        # 先将所有视觉位置设为 False
        global_keep_mask[is_visual] = False
        
        # 找出需要保留的全局索引
        # keep_mask_visual 是对应 visual_indices_global 的 mask
        indices_to_keep_global = visual_indices_global[keep_mask_visual]
        
        # 将保留的位置设为 True
        global_keep_mask[indices_to_keep_global] = True
        
        # ================= [保留] Debug / 插桩代码 =================
        if not hasattr(self, "debug_info"): self.debug_info = {}
        
        # 为了 debug 方便，我们需要提取被保留下来的 visual token 的原始坐标
        # position_ids shape: [3, BS, Seq_Len], 我们取 [1,0] (H) 和 [2,0] (W)
        vis_pos_h = position_ids[1, 0, is_visual] 
        vis_pos_w = position_ids[2, 0, is_visual]
        
        # 记录被保留的视觉 Token 的 mask (在视觉序列内部的 mask)
        self.debug_info["last_mask"] = keep_mask_visual.cpu().numpy()
        self.debug_info["last_h"] = vis_pos_h.cpu().numpy()
        self.debug_info["last_w"] = vis_pos_w.cpu().numpy()
        # =========================================================

        # --- 6. 应用剪枝 ---
        # 剪枝 hidden_states: [BS, Seq_Len, Hidden] -> [BS, New_Len, Hidden]
        pruned_hidden_states = hidden_states[:, global_keep_mask, :]
        
        # 剪枝 position_ids: [3, BS, Seq_Len] -> [3, BS, New_Len]
        pruned_position_ids = position_ids[:, :, global_keep_mask]
        
        # 剪枝 attention_mask: [BS, 1, Seq_Len, Seq_Len] -> [BS, 1, New_Len, New_Len]
        # 注意：Attention Mask 通常是正方形或者特定形状，这里假设是标准的 causal mask 变体
        if attention_mask is not None:
             # 我们同时对 Q 和 K 维度进行剪枝，保持 mask 为方阵（针对 causal decoder）
             pruned_attention_mask = attention_mask[:, global_keep_mask]
        else:
            pruned_attention_mask = None

        return pruned_hidden_states, pruned_position_ids, pruned_attention_mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, cache_position=None):
        
        if input_ids is None and hasattr(self, "temporary_input_ids"):
            input_ids = self.temporary_input_ids

        FASTV_k = 2 
        layer_config = {
            "FASTV_r": 0.5,
            "rank_fusion_weight": 0.5,
            "WIN_H": 4, 
            "WIN_W": 4 
        }

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            layer_idx = decoder_layer.self_attn.layer_idx

            temp_output_attentions = output_attentions
            if layer_idx == FASTV_k - 1:
                temp_output_attentions = True

            if layer_idx == FASTV_k and self.last_attention is not None:
                if hidden_states.shape[1] > 1:
                    hidden_states, position_ids, attention_mask = self._prune_visual_tokens(
                        hidden_states, position_ids, attention_mask, input_ids, layer_config
                    )
                    
                    if cache_position is not None:
                        cache_position = cache_position[:hidden_states.shape[1]]
                    
                    causal_mask = self._update_causal_mask(
                         attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                    )
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=temp_output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if temp_output_attentions else 1]

            if layer_idx == FASTV_k - 1:
                self.last_attention = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class FastDCQwen2_5_VLForConditionalGeneration(OriginalForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = FastDCQwen2_5_VLModel(config)
        self.post_init()
        
    def forward(self, input_ids=None, **kwargs):
        if hasattr(self, "model") and input_ids is not None:
            self.model.temporary_input_ids = input_ids
        return super().forward(input_ids=input_ids, **kwargs)

# ==============================================================================
# 2. 工具函数
# ==============================================================================

def get_test_inputs(processor, device, height=1024, width=1024):
    image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe."}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs.to(device)

def get_cache_size(past_key_values):
    if past_key_values is None: return 0
    total_bytes = 0
    if hasattr(past_key_values, "key_cache"):
        for k in past_key_values.key_cache:
            total_bytes += k.element_size() * k.nelement()
        for v in past_key_values.value_cache:
            total_bytes += v.element_size() * v.nelement()
    return total_bytes / (1024**3)

def run_benchmark(model, inputs, max_new_tokens=50, name="Model", num_runs=5):
    """
    修改版：增加 num_runs 参数，进行多次测试取平均值
    """
    print(f"\n>>> Benchmarking: {name} (Averaging over {num_runs} runs)")
    device = model.device
    model.eval()
    
    # === 1. Warmup (热身) ===
    # 热身非常重要，确保 GPU 进入高性能状态，且 CUDA context 初始化完毕
    print("  Warmup (3 runs)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=2, use_cache=True)
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 定义列表存储每次的结果
    prefill_times = []
    decoding_latencies = []
    peak_memories = []
    
    # === 2. 循环测试 ===
    for i in range(num_runs):
        print(f"  Running test {i+1}/{num_runs}...", end="\r")
        
        # --- A. 测试 Prefill ---
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # 强制同步，确保计时准确
        torch.cuda.synchronize() 
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            
        torch.cuda.synchronize()
        p_time = time.time() - start_time
        prefill_times.append(p_time)
        
        # 记录显存 (取多次运行中的最大值作为峰值，或者平均值都可以，通常取最大值更安全)
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        peak_memories.append(mem)
        
        # --- B. 测试 Decoding ---
        # 注意：为了准确，这里重新进行一次完整的 generate
        # 我们利用刚才 measured 的 p_time 来从总时间里扣除
        
        torch.cuda.synchronize()
        start_gen = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        torch.cuda.synchronize()
        total_time = time.time() - start_gen
        
        # 计算纯解码时间
        d_time = total_time - p_time
        # 防止极端波动导致负数
        if d_time < 0: d_time = 0 
        
        # 计算每个 token 的延迟
        avg_lat = (d_time / max_new_tokens) * 1000 # ms
        decoding_latencies.append(avg_lat)

    print(f"  Running test {num_runs}/{num_runs}... Done.")

    # === 3. 计算统计结果 ===
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decoding = sum(decoding_latencies) / len(decoding_latencies)
    avg_memory = sum(peak_memories) / len(peak_memories)
    
    # 计算 Cache Size (只需要取最后一次的就行，因为结构不变)
    cache_size = get_cache_size(outputs.past_key_values)

    # 打印剪枝信息
    if hasattr(model.model, "debug_info") and "pruned_count" in model.model.debug_info:
        info = model.model.debug_info
        print(f"  [Info] Visual Tokens: {info['total_visual']} -> Pruned: {info['pruned_count']} "
              f"({info['pruned_count']/info['total_visual']:.1%})")
    
    results = {
        "Peak Memory (GB)": avg_memory,
        "Prefill Time (s)": avg_prefill,
        "Decoding Latency (ms)": avg_decoding,
        "Cache Size (GB)": cache_size
    }
    
    # 可以在这里打印标准差 (Std Dev)，论文中加上这个会显得更专业
    prefill_std = np.std(prefill_times)
    decoding_std = np.std(decoding_latencies)
    print(f"  Results (Avg): {results}")
    print(f"  Std Dev: Prefill ±{prefill_std:.4f}s, Decoding ±{decoding_std:.2f}ms")
    
    return results
# ==============================================================================
# 3. 主程序 (修改为串行加载模式)
# ==============================================================================

def main():
    model_path = "/root/final_paper/qwen2_5_vl_3b"
    device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"Loading processor from {model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1400*1400)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    print("Generating test inputs...")
    inputs = get_test_inputs(processor, device, height=700, width=70)
    print(f"Input shape (Input IDs): {inputs.input_ids.shape}")

    # ---------------------------
    # 第一阶段：加载原版 -> 测试 -> 删除
    # ---------------------------
    print("\n" + "="*40)
    print(" PHASE 1: Original Model")
    print("="*40)
    print("Loading Original Qwen2.5-VL (Eager Mode)...")
    
    model = OriginalForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager", # 修正：Eager模式
        device_map=device
    )
    stats_orig = run_benchmark(model, inputs, max_new_tokens=50, name="Original")
    
    # 关键步骤：彻底清理内存，防止 OOM
    print("Cleaning up Original Model to free memory...")
    del model
    gc.collect()             # 强制运行 Python 垃圾回收
    torch.cuda.empty_cache() # 释放显存
    time.sleep(2)            # 给系统一点缓冲时间

    # ---------------------------
    # 第二阶段：加载 FastDC 版 -> 测试
    # ---------------------------
    print("\n" + "="*40)
    print(" PHASE 2: FastDC Algorithm")
    print("="*40)
    print("Loading FastDC Model (from disk)...")
    
    # 修正：直接从磁盘加载权重到 FastDC 类
    # 因为 FastDC 类继承自原版且结构一致，load_weights 会自动匹配
    fast_model = FastDCQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager", # 修正：Eager模式
        device_map=device
    )
    
    stats_fast = run_benchmark(fast_model, inputs, max_new_tokens=50, name="FastDC")

    # ---------------------------
    # 结果对比
    # ---------------------------
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'Original':<12} | {'FastDC':<12} | {'Diff':<10}")
    print("-" * 60)
    for k in stats_orig.keys():
        v1 = stats_orig[k]
        v2 = stats_fast[k]
        diff = (v2 - v1) / v1 * 100
        print(f"{k:<25} | {v1:<12.4f} | {v2:<12.4f} | {diff:+.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()

