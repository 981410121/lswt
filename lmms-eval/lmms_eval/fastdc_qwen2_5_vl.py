import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math
from transformers.cache_utils import DynamicCache
# 引用上传文件中的 Qwen2.5-VL 模块
from .qwen2_5_vl_4_51_3.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    BaseModelOutputWithPast
)
from .qwen2_5_vl_4_51_3.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class FastDCQwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.last_attention = None 
        # 用于暂存 Wrapper 传进来的 input_ids
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
        执行 FastDC 剪枝逻辑 (适配 Qwen2.5-VL 动态分辨率)
        """
        # --- 安全检查 ---
        if input_ids is None:
            # 如果依然没拿到 input_ids，无法定位视觉 token，只能跳过剪枝
            return hidden_states, position_ids, attention_mask

        FASTV_r = layer_config.get("FASTV_r", 0.5)
        rank_fusion_weight = float(layer_config.get("rank_fusion_weight", 0.5))
        WIN_H = layer_config.get("WIN_H", 8) 
        WIN_W = layer_config.get("WIN_W", 8)
        
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        # import ipdb
        # ipdb.set_trace()
        # --- 兼容性检查 ---
        if bsz > 1:
            return hidden_states, position_ids, attention_mask
            
        # 1. 识别视觉 Token
        image_token_id = self.config.image_token_id
        
        # 确保 input_ids 维度正确
        if input_ids.shape[0] != bsz:
             # 极端情况处理
             return hidden_states, position_ids, attention_mask

        is_visual = (input_ids[0] == image_token_id) # [Seq_Len]
        
        if not is_visual.any():
            return hidden_states, position_ids, attention_mask

        visual_indices = torch.where(is_visual)[0]
        
        # 2. 获取分数
        if self.last_attention is None:
             return hidden_states, position_ids, attention_mask

        attn_weights = torch.mean(self.last_attention, dim=1) # [BS, Q, K]
        attn_img = attn_weights[0, :, is_visual] # [Seq_Len, Num_Visual]
        
        n_seq, n_img = attn_img.shape
        if n_img == 0: return hidden_states, position_ids, attention_mask

        # 计算 High-Pass Score (Freq)
        a_lp = 1.0 / n_img
        a_hp = attn_img - a_lp
        score_freq_raw = torch.sum(a_hp, dim=0) / n_seq 
        
        # 计算 FastV Score
        score_fastv_raw = attn_img[-1, :] 

        # 3. 动态恢复 Grid
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

        # 4. 窗口遍历与剪枝
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

        # 5. 应用剪枝
        global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        global_keep_mask[is_visual] = keep_mask_visual
        # ================= [新增] 插桩代码 =================
        # 将本次推理的 掩码 和 原始位置信息 存到 self 中供外部调用
        if not hasattr(self, "debug_info"): self.debug_info = {}
        self.debug_info["last_mask"] = keep_mask_visual.cpu().numpy() # 只存视觉部分的 mask
        self.debug_info["last_h"] = vis_pos_h.cpu().numpy()           # 原始 H 坐标
        self.debug_info["last_w"] = vis_pos_w.cpu().numpy()           # 原始 W 坐标
        # ===================================================
        # import ipdb
        # ipdb.set_trace()
        pruned_hidden_states = hidden_states[:, global_keep_mask, :]
        pruned_position_ids = position_ids[:, :, global_keep_mask]
        
        if attention_mask is not None:
            pruned_attention_mask = attention_mask[:, global_keep_mask]
        else:
            pruned_attention_mask = None

        return pruned_hidden_states, pruned_position_ids, pruned_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # === 【核心修复】尝试找回 input_ids ===
        if input_ids is None and hasattr(self, "temporary_input_ids"):
            input_ids = self.temporary_input_ids
        # ===================================

        FASTV_k = 2 
        layer_config = {
            "FASTV_r": 0.66,
            "rank_fusion_weight": 0,
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

            # === FastDC 剪枝逻辑 ===
            if layer_idx == FASTV_k and self.last_attention is not None:
                if hidden_states.shape[1] > 1:
                    debug_len_before = hidden_states.shape[1]
                    # import ipdb
                    # ipdb.set_trace()
                    hidden_states, position_ids, attention_mask = self._prune_visual_tokens(
                        hidden_states, position_ids, attention_mask, input_ids, layer_config
                    )
                    debug_len_after = hidden_states.shape[1]
                    # [建议] 直接在这里打印，比外部 Hook 更靠谱
                    # print(f">>> Layer {layer_idx} Pruning: {debug_len_before} -> {debug_len_after}")
                    # 剪枝后更新 meta data
                    if cache_position is not None:
                        cache_position = cache_position[:hidden_states.shape[1]]
                    
                    causal_mask = self._update_causal_mask(
                         attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                    )
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # === Layer Forward ===
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    temp_output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
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

class FastDCQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        fast_model = FastDCQwen2_5_VLModel(config)
        fast_model.load_state_dict(self.model.state_dict())
        self.model = fast_model
        
    def forward(self, input_ids=None, **kwargs):
        # 【中转传值】将 input_ids 暂存到内部模型，解决父类吞掉参数的问题
        if hasattr(self, "model") and input_ids is not None:
            self.model.temporary_input_ids = input_ids
            
        return super().forward(input_ids=input_ids, **kwargs)