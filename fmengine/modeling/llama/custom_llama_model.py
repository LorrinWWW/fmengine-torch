import torch
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, logging, replace_return_docstrings
from transformers import LlamaConfig

from fmengine.modeling._common._nn import EmbeddingPipe, LMLayerPipe
from fmengine import mpu

try:
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
    flash_attn_v2_installed = True
    from flash_attn.layers.rotary import RotaryEmbedding as _RotaryEmbedding
    from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

    class RotaryEmbedding(_RotaryEmbedding):
        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            scaling_factor=1.0,
            pos_idx_in_fp32=True,
            device=None,
        ):
            self.scaling_factor = scaling_factor
            super().__init__(
                dim=dim,
                base=base,
                interleaved=interleaved,
                scale_base=scale_base,
                pos_idx_in_fp32=pos_idx_in_fp32,
                device=device,
            )
    
        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            ):
                self._seq_len_cached = seqlen
                if self.pos_idx_in_fp32:
                    t = torch.arange(seqlen, device=device, dtype=torch.float32)
                    # linear interpolation
                    t /= self.scaling_factor
                    if self.inv_freq.dtype != torch.float32:
                        inv_freq = self._compute_inv_freq(device=device)
                    else:
                        inv_freq = self.inv_freq
                else:
                    t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                    # linear interpolation
                    t /= self.scaling_factor
                    inv_freq = self.inv_freq
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = (
                        torch.arange(
                            seqlen, dtype=self.scale.dtype, device=self.scale.device
                        )
                        - seqlen // 2
                    ) / self.scale_base
                    scale = self.scale.to(device=power.device) ** rearrange(
                        power, "s -> s 1"
                    )
                    # We want the multiplication by scale to happen in fp32
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)
except ImportError:
    flash_attn_v2_installed = False
    print('Warn: Error when importing flash attention.')

# def rmsnorm_func(hidden_states, weight, variance_epsilon):
#     input_dtype = hidden_states.dtype
#     hidden_states = hidden_states.to(torch.float32)
#     variance = hidden_states.pow(2).mean(-1, keepdim=True)
#     hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
#     return weight * hidden_states.to(input_dtype)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "variance_epsilon",
            torch.tensor(eps),
            persistent=False,
        )

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)

class LastRMSNorm(RMSNorm):
    def forward(self, fw_args):
        hidden_states, *_ = fw_args
        hidden_states = rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)
        return (hidden_states,)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        args,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.up_proj = mpu.ColumnParallelLinear(
        #     args=args.deepspeed_config,
        #     input_size=hidden_size,
        #     output_size=intermediate_size,
        #     gather_output=False,
        #     init_method=nn.init.xavier_normal_,
        #     skip_bias_add=True,
        #     bias=False,
        # )
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        args,
        config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        max_positions = config.max_position_embeddings
        self.max_positions = max_positions
        self.config = config

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.rotary_ndims = self.head_dim
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )

        if self.config.rope_scaling is None:
            # by default do linear scale if not specified.
            scaling_factor = max(self.max_positions / 4096, 1.0)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            assert scaling_type == "linear"
        
        print(f"Linearly scaling {scaling_factor}x.")
        
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=config.rope_theta,
            interleaved=False,
            scaling_factor=scaling_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        query_states = (self.q_proj(hidden_states)).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        value_states = (self.v_proj(hidden_states)).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )

        q = query_states
        kv = torch.stack([key_states, value_states], dim=2)
        q, kv = self.rotary_emb(q, kv)

        if flash_attn_v2_installed:
            attn_output = flash_attn_kvpacked_func(
                q, kv, 0.0,
                causal=True,
            )
        else:
            raise Exception("Flash Attention not found.")

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
                

class ParallelTransformerLayerPipe(nn.Module):
    def __init__(self, args, config: LlamaConfig, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(args=args, config=config)
        self.mlp = LlamaMLP(
            args=args,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        def attn_res(hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states
            return hidden_states

        self.attn_res = attn_res

        def mlp_res(hidden_states: torch.Tensor) -> torch.Tensor:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        self.mlp_res = mlp_res
        

    def forward(self, fw_args):

        x, position_ids, mask = fw_args
        attention_mask = None

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.attn_res, x, attention_mask)
        else:
            x = self.attn_res(x, attention_mask)

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)

        return (x, position_ids, mask)


class CustomLlamaModelPipe(PipelineModule):
    def __init__(self, args, model_config, activation_checkpointing_config, **kwargs):
        if activation_checkpointing_config:
            deepspeed.checkpointing.configure(
                mpu,
                partition_activations=activation_checkpointing_config.get(
                    "partition_activations", False
                ),
                contiguous_checkpointing=activation_checkpointing_config.get(
                    "contiguous_memory_optimization", False
                ),
                checkpoint_in_cpu=activation_checkpointing_config.get(
                    "cpu_checkpointing", False
                ),
                num_checkpoints=activation_checkpointing_config.get(
                    "number_checkpoints", None
                ),
                synchronize=activation_checkpointing_config.get(
                    "synchronize_checkpoint_boundary", False
                ),
                profile=activation_checkpointing_config.get("profile", False),
            )
        super().__init__(
            layers=[
                LayerSpec(
                    EmbeddingPipe,
                    model_config.vocab_size,
                    model_config.hidden_size,
                ),
                *[
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        args,
                        model_config,
                        activation_checkpointing_config is not None,
                    )
                    for _ in range(model_config.num_hidden_layers)
                ],
                LayerSpec(
                    LastRMSNorm,
                    model_config.hidden_size,
                    model_config.rms_norm_eps,
                ),
                LayerSpec(
                    LMLayerPipe,
                    model_config.hidden_size,
                    model_config.vocab_size,
                    bias=False,
                ),
            ],
            **kwargs
        )