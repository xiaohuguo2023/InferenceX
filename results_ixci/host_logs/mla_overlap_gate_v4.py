# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD-specific MLA wrapper for Kimi-K3."""

from typing import cast

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.kimi_k3.amd.ops.sigmoid_gate import sigmoid_gate_mul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


class KimiK3MultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """Kimi-K3 MLA wrapper with eager AITER q/kv RMSNorm fusion."""

    def __init__(
        self,
        *args,
        gate_stream: torch.cuda.Stream | None = None,
        gate_stream_token_threshold: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_eager_qk_rmsnorm_fusion = bool(rocm_aiter_ops.is_enabled())
        self._gate_stream = gate_stream if self.g_proj is not None else None
        self._gate_stream_token_threshold = gate_stream_token_threshold
        self._gate_events = (
            (torch.cuda.Event(), torch.cuda.Event())
            if self._gate_stream is not None
            else None
        )

    def _normalize_q_kv(
        self,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_layernorm = cast(RMSNorm, self.q_a_layernorm)
        kv_layernorm = cast(RMSNorm, self.kv_a_layernorm)

        if self._use_eager_qk_rmsnorm_fusion and not torch.compiler.is_compiling():
            return torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c,
                q_layernorm.weight,
                kv_c,
                kv_layernorm.weight,
                q_layernorm.variance_epsilon,
                kv_layernorm.variance_epsilon,
            )

        return q_layernorm(q_c), kv_layernorm(kv_c)

    def _forward_attention_frontend(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_proj_input, kv_c_normed = self._normalize_q_kv(q_c, kv_c)
            q_proj_layer = self.q_b_proj
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        # Add head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
        )
        return attn_out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g_proj = self.g_proj
        events = self._gate_events
        stream = self._gate_stream
        use_stream = (
            g_proj is not None
            and events is not None
            and stream is not None
            and hidden_states.shape[0] <= self._gate_stream_token_threshold
        )
        if use_stream:
            assert g_proj is not None
            assert events is not None
            assert stream is not None
            attn_out, (gate, _) = maybe_execute_in_parallel(
                lambda: self._forward_attention_frontend(
                    positions, hidden_states, llama_4_scaling
                ),
                lambda: g_proj(hidden_states),
                events[0],
                events[1],
                stream,
            )
        else:
            attn_out = self._forward_attention_frontend(
                positions, hidden_states, llama_4_scaling
            )
            gate = g_proj(hidden_states)[0] if g_proj is not None else None

        if gate is not None:
            attn_out = sigmoid_gate_mul(attn_out, gate)
        return self.o_proj(attn_out)[0]
