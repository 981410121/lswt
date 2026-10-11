import torch
import time
import os
import gc
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

    def _prune_visual_tokens(self, hidden_states, position_ids, attention_mask, input_ids, layer_config):
        import math
        # --- 安全检查 ---
        if input_ids is None: return hidden_states, position_ids, attention_mask

        FASTV_r = layer_config.get("FASTV_r", 0.5)
        rank_fusion_weight = float(layer_config.get("rank_fusion_weight", 0.5))
        WIN_H = layer_config.get("WIN_H", 8) 
        WIN_W = layer_config.get("WIN_W", 8)
        
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        if bsz > 1: return hidden_states, position_ids, attention_mask
            
        image_token_id = self.config.image_token_id
        if input_ids.shape[0] != bsz: return hidden_states, position_ids, attention_mask

        is_visual = (input_ids[0] == image_token_id)
        if not is_visual.any(): return hidden_states, position_ids, attention_mask

        visual_indices = torch.where(is_visual)[0]
        
        if self.last_attention is None: return hidden_states, position_ids, attention_mask

        attn_weights = torch.mean(self.last_attention, dim=1) 
        attn_img = attn_weights[0, :, is_visual] 
        
        n_seq, n_img = attn_img.shape
        if n_img == 0: return hidden_states, position_ids, attention_mask

        a_lp = 1.0 / n_img
        a_hp = attn_img - a_lp
        score_freq_raw = torch.sum(a_hp, dim=0) / n_seq 
        score_fastv_raw = attn_img[-1, :] 

        vis_pos_h = position_ids[1, 0, is_visual] 
        vis_pos_w = position_ids[2, 0, is_visual] 

        unique_h = torch.unique(vis_pos_h, sorted=True)
        unique_w = torch.unique(vis_pos_w, sorted=True)

        grid_h_indices = torch.searchsorted(unique_h, vis_pos_h)
        grid_w_indices = torch.searchsorted(unique_w, vis_pos_w)

        GRID_H_SIZE = len(unique_h)
        GRID_W_SIZE = len(unique_w)

        num_win_h = math.ceil(GRID_H_SIZE / WIN_H)
        num_win_w = math.ceil(GRID_W_SIZE / WIN_W)

        keep_mask_visual = torch.zeros_like(vis_pos_h, dtype=torch.bool, device=device)

        for gh in range(num_win_h):
            for gw in range(num_win_w):
                h_start = gh * WIN_H
                h_end = min((gh + 1) * WIN_H, GRID_H_SIZE)
                w_start = gw * WIN_W
                w_end = min((gw + 1) * WIN_W, GRID_W_SIZE)
                
                in_window_mask = (
                    (grid_h_indices >= h_start) & (grid_h_indices < h_end) &
                    (grid_w_indices >= w_start) & (grid_w_indices < w_end)
                )
                
                window_indices = torch.where(in_window_mask)[0]
                if len(window_indices) == 0: continue

                local_freq_raw = score_freq_raw[window_indices] 
                local_fastv_raw = score_fastv_raw[window_indices]
                
                _, local_freq_sorted_idx = torch.sort(local_freq_raw, descending=True)
                _, local_fastv_sorted_idx = torch.sort(local_fastv_raw, descending=True)
                
                num_tokens_in_window = len(window_indices)
                local_weighting = torch.arange(num_tokens_in_window, 0, -1, device=device, dtype=torch.float32)
                
                local_rank_score = torch.zeros(num_tokens_in_window, device=device)
                local_rank_score[local_freq_sorted_idx] += local_weighting * rank_fusion_weight
                local_rank_score[local_fastv_sorted_idx] += local_weighting * (1 - rank_fusion_weight)
                
                local_keep_k = int(num_tokens_in_window * (1 - FASTV_r))
                
                if local_keep_k > 0:
                    local_top_k_indices_in_window = local_rank_score.topk(local_keep_k).indices
                    global_indices_to_keep = window_indices[local_top_k_indices_in_window]
                    keep_mask_visual[global_indices_to_keep] = True

        global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        global_keep_mask[is_visual] = keep_mask_visual
        
        if not hasattr(self, "debug_info"): self.debug_info = {}
        self.debug_info["pruned_count"] = (~keep_mask_visual).sum().item()
        self.debug_info["total_visual"] = len(keep_mask_visual)

        pruned_hidden_states = hidden_states[:, global_keep_mask, :]
        pruned_position_ids = position_ids[:, :, global_keep_mask]
        
        if attention_mask is not None:
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

# ==============================================================================
# 3. 性能测试主流程 (【核心修改】：增加 processor 参数并打印输出)
# ==============================================================================

def run_benchmark(model, inputs, processor, max_new_tokens=50, name="Model", num_runs=5):
    print(f"\n>>> Benchmarking: {name} (Averaging over {num_runs} runs)")
    device = model.device
    model.eval()
    
    # === 1. Warmup ===
    print("  Warmup (1 runs)...")
    for _ in range(1):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=2, use_cache=True)
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    prefill_times = []
    decoding_latencies = []
    peak_memories = []
    
    # 存储最后一次生成的 Token ID 用于打印
    final_output_ids = None

    # === 2. 循环测试 ===
    for i in range(num_runs):
        print(f"  Running test {i+1}/{num_runs}...", end="\r")
        
        # --- A. 测试 Prefill ---
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            
        torch.cuda.synchronize()
        p_time = time.time() - start_time
        prefill_times.append(p_time)
        
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        peak_memories.append(mem)
        
        # --- B. 测试 Decoding 及其输出 ---
        torch.cuda.synchronize()
        start_gen = time.time()
        with torch.no_grad():
            # 【核心修改】这里捕获输出 generated_ids
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        torch.cuda.synchronize()
        total_time = time.time() - start_gen
        
        # 保存最后一次的输出 ID
        if i == num_runs - 1:
            final_output_ids = generated_ids
        
        d_time = total_time - p_time
        if d_time < 0: d_time = 0 
        avg_lat = (d_time / max_new_tokens) * 1000 # ms
        decoding_latencies.append(avg_lat)

    print(f"  Running test {num_runs}/{num_runs}... Done.")

    # === 3. 计算统计结果 ===
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decoding = sum(decoding_latencies) / len(decoding_latencies)
    avg_memory = sum(peak_memories) / len(peak_memories)
    cache_size = get_cache_size(outputs.past_key_values)

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
    
    prefill_std = np.std(prefill_times)
    decoding_std = np.std(decoding_latencies)
    print(f"  Results (Avg): {results}")
    print(f"  Std Dev: Prefill ±{prefill_std:.4f}s, Decoding ±{decoding_std:.2f}ms")

    # === 4. 【核心修改】打印模型生成内容 ===
    if final_output_ids is not None:
        # 去掉输入部分的 token，只看生成的 token
        # input_ids 的长度
        input_len = inputs.input_ids.shape[1]
        # 截取
        generated_ids_trimmed = [
            out_ids[input_len:] for out_ids in final_output_ids
        ]
        # 解码
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("\n  [Generated Output Preview]:")
        print(f"  > {output_text[0]}")
    
    return results

# ==============================================================================
# 4. 主程序
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
    inputs = get_test_inputs(processor, device, height=840, width=840)
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
        attn_implementation="eager", 
        device_map=device
    )
    # 【核心修改】传入 processor
    stats_orig = run_benchmark(model, inputs, processor=processor, max_new_tokens=500, name="Original")
    
    print("Cleaning up Original Model to free memory...")
    del model
    gc.collect()            
    torch.cuda.empty_cache() 
    time.sleep(2)            

    # ---------------------------
    # 第二阶段：加载 FastDC 版 -> 测试
    # ---------------------------
    print("\n" + "="*40)
    print(" PHASE 2: FastDC Algorithm")
    print("="*40)
    print("Loading FastDC Model (from disk)...")
    
    fast_model = FastDCQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager",
        device_map=device
    )
    
    # 【核心修改】传入 processor
    stats_fast = run_benchmark(fast_model, inputs, processor=processor, max_new_tokens=500, name="FastDC")

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