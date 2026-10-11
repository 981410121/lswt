import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math


from qwen2_5_vl_4_51_3.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    BaseModelOutputWithPast
)
from qwen2_5_vl_4_51_3.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class FastVQwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config: Qwen2_5_VLConfig):
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

        # === FastV 配置 ===
        # FastV 通常只在某一层做一次剪枝 (K)，通常是较浅的层（如第2层或第3层）
        FASTV_k = 2 
        layer_config = {
            "FASTV_r": 0.5, # 剪枝比例，0.5 代表丢弃 50%
        }

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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
            # 如果是 K-1 层，我们需要强制开启 output_attentions 以便下一层(K层)拿到 attention score
            if layer_idx == FASTV_k - 1:
                temp_output_attentions = True

            # === FastV 剪枝执行点 ===
            if layer_idx == FASTV_k and self.last_attention is not None:
                # 只有当序列长度大于1 (且包含视觉信息) 时才尝试剪枝
                if hidden_states.shape[1] > 1:
                    debug_len_before = hidden_states.shape[1]
                    
                    hidden_states, position_ids, attention_mask = self._prune_visual_tokens(
                        hidden_states, position_ids, attention_mask, input_ids, layer_config
                    )
                    import ipdb; ipdb.set_trace()
                    debug_len_after = hidden_states.shape[1]
                    if debug_len_after < debug_len_before:
                        print(f">>> FastV Layer {layer_idx} Pruning: {debug_len_before} -> {debug_len_after}")
                    
                    # 剪枝后必须更新 meta data，否则后续计算会报错
                    if cache_position is not None:
                        # 截断 cache_position 以匹配新的长度
                        cache_position = cache_position[:hidden_states.shape[1]]
                    
                    # 重新生成 mask 和 embedding
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

            # 存储 K-1 层的 attention 给 K 层使用
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

# 对应的 ConditionalGeneration Wrapper
class FastVQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        fast_model = FastVQwen2_5_VLModel(config)
        # 加载权重 (假设父类已经加载了)
        fast_model.load_state_dict(self.model.state_dict(), strict=False)
        self.model = fast_model
        
    def forward(self, input_ids=None, **kwargs):
        # 【中转传值】将 input_ids 暂存到内部模型
        if hasattr(self, "model") and input_ids is not None:
            self.model.temporary_input_ids = input_ids
            
        return super().forward(input_ids=input_ids, **kwargs)