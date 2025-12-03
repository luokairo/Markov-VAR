from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from typing import Tuple, Union
from models.utils import DropPath

from torch.nn.functional import scaled_dot_product_attention as slow_attn

dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_kvpacked_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, falling back to standard attention")


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()

class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # TODO: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        
        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens
        
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


class ContextAttentivePool(nn.Module):
    def __init__(self, C: int, D):
        super().__init__()
        self.C, self.D = C, D
        if D > 4096:
            self.head_dim = 64
        else:
            self.head_dim = 128

        self.num_heads = C // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=self.C, num_heads=self.num_heads)

    @staticmethod
    def _pack_varlen_kv(x_BLC: torch.Tensor, valid_lens: torch.Tensor):
        device = x_BLC.device
        B, Lmax, C = x_BLC.shape
        lens = valid_lens.tolist()

        slices = [x_BLC[i, :lens[i], :] for i in range(B)]
        kv_compact = torch.cat(slices, dim=0)  # (ΣL_i, C)

        cu_seqlens_k = torch.tensor(
            [0] + [sum(lens[:i+1]) for i in range(B)],
            device=device, dtype=torch.int32
        )

        max_seqlen_k = int(max(lens))
        return kv_compact, cu_seqlens_k, max_seqlen_k

    def forward(self, ca_kv): 
        B, Lw, C = ca_kv.shape
        valid_lens = torch.full((B,), Lw, dtype=torch.int32, device=ca_kv.device)
        kv_compact, cu_seqlens_k, max_seqlen_k = self._pack_varlen_kv(ca_kv, valid_lens)
        ca_kv_tuple = (kv_compact, cu_seqlens_k, max_seqlen_k)

        pooled = self.ca(None, ca_kv_tuple)  # (B, 1, D)
        return pooled.squeeze(1)             # (B, D)


@functools.cache
def get_position_ids(batch_size, patch_nums, device, si=-1):
    all_position_ids = []
    largest_patch_num = patch_nums[-1]
    if si == -1:
        pns = patch_nums
    else:
        pns = patch_nums[si: si+1]
    for level_idx in range(len(pns)):
        patch_num = pns[level_idx]
        _x = torch.arange(patch_num, device=device)
        _y = torch.arange(patch_num, device=device)
        # [pn, pn, 2]
        cartesian = torch.stack(torch.meshgrid(_x, _y, indexing="ij"), dim=-1)
        # normalize to the size in the largest feature map
        coords = cartesian / patch_num * largest_patch_num
        # [pn * pn, 2]
        coords = coords.reshape(-1, 2)
        all_position_ids.append(coords)

    pos_ids = torch.cat(all_position_ids, 0).unsqueeze(0).repeat(batch_size, 1, 1)
    return pos_ids

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 4]
        # inv_freq_expanded: [bs, head_size // 4, 1, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None, None]
            .float()
            .expand(position_ids.shape[0], -1, 1, 1)
            .repeat(1, 1, 1, 2)
        )
        # position_ids_expanded: [bs, 1, seq_len, 2]
        position_ids_expanded = position_ids[:, None, :].float()
        inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
        position_ids_expanded = position_ids_expanded.permute(0, 3, 1, 2).contiguous()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, 2, seq_len, head_size // 4]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, 2, seq_len, head_size // 2]
            cos = emb.cos()
            sin = emb.sin()
            # [bs, seq_len, 2, head_size // 2]
            cos = cos.transpose(2, 1).contiguous()
            sin = sin.transpose(2, 1).contiguous()
            cos = cos.reshape(cos.size(0), cos.size(1), -1)
            sin = sin.reshape(sin.size(0), sin.size(1), -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding1D(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 2]
        # inv_freq_expanded: [bs, head_size // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, seq_len, head_size // 2]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, seq_len, head_size]
            cos = emb.cos()
            sin = emb.sin()
        # [bs, seq_len, head_size]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        flag = False
        if len(t.shape) == 2:
            flag = True
            t = t[0]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.dtype
        )
        t_emb = self.mlp(t_freq)
        if not flag:
            return t_emb
        else:
            return t_emb.unsqueeze(0)

    @property
    def dtype(self):
        # return the data type of this model
        return next(self.parameters()).dtype

class LlamaAttention(nn.Module):
    def __init__(
        self, 
        block_idx,
        patch_nums,
        embed_dim=1024,
        num_heads=16,
        attn_drop=0.0,
        proj_drop=0.0,
        max_position_embeddings=4096,
        rope_theta=10000,
        flash_if_available=True,
        attn_l2_norm=False,
        context_token=0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert patch_nums is not None
        self.context_token = context_token
        self.patch_nums = patch_nums
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attn_l2_norm = attn_l2_norm

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=self.rope_theta
        )

        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(
            torch.zeros(embed_dim)
        )
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            False  # flash_if_available and memory_efficient_attention is not None
        )

        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        attn_bias,
        si=-1,
    ):
        B, L, C = x.shape
        if self.context_token == 1:
            position_ids = get_position_ids(
                B, self.patch_nums, x.device, si,
            )
        else:
            raise ValueError(
            f"Unsupported context_token={self.context_token}. "
            "Expected 0 (no context) or 1 (single global token). "
            "If using text-to-image mode, please set context_token>1 "
            "and provide valid context_position_ids."
        )
    
        qkv = F.linear(
            input=x,
            weight=self.qkv_proj.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
            dim_unsqueeze = 2
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc
            dim_unsqueeze = 1

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.context_token == 1:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            print("Context token cannot be negative", self.context_token)
            raise NotImplementedError
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            # raise NotImplementedError
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LlamaMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.gate_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.up_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.down_proj = nn.Linear(self.hidden_features, self.out_features, bias=False)
        self.act_fn = nn.SiLU()
        self.fused_mlp_func = None

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaAdaLNSelfAttn(nn.Module):
    def __init__(
        self, 
        block_idx,
        norm_layer,
        last_drop_p,
        embed_dim=768,
        cond_dim=768,
        shared_aln=False,
        num_heads=16,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
        attn_l2_norm=False,
        flash_if_available=False,
        fused_if_available=True,
        disable_aln=False,
        max_position_embeddings=4096,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        context_token=1,
        rope_theta=10000,
        use_cross_attn=False,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.disable_aln = disable_aln

        self.attn = LlamaAttention(
            block_idx=block_idx,
            patch_nums=patch_nums,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            flash_if_available=flash_if_available,
            attn_l2_norm=attn_l2_norm,
            context_token=context_token
        )

        self.ffn = LlamaMLP(
            in_features=embed_dim,
            hidden_features=int((embed_dim * mlp_ratio * 2) / 3 + 255) // 256 * 256,
            out_features=embed_dim,
        )

        self.ln_wo_grad = norm_layer(embed_dim)
        self.shared_aln = shared_aln
        if not self.disable_aln:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        else:
            if self.shared_aln:
                self.scale_shift_table = nn.Parameter(
                    torch.randn(6, embed_dim) / embed_dim**0.5
                )

        self.use_cross_attn = use_cross_attn
        self.fused_add_norm_fn = None
        if self.use_cross_attn:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
        else:
            self.cross_attn = None

    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        si=-1,
    ):
        if not self.disable_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
            )
            x = x + self.drop_path(
                self.attn(
                    self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                    attn_bias=attn_bias,
                    si=si
                ).mul_(gamma1)
            )
            if self.use_cross_attn:
                x[:, cond_BD.size(1) :] += self.cross_attn(
                    x[:, cond_BD.size(1) :], cond_BD, None
                )
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            ) 
        else:
            if not self.shared_aln:
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x),
                        attn_bias=attn_bias,
                        si=si,
                    )
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))
            else:
                adaln_modulator = self.scale_shift_table[None] + cond_BD.unsqueeze(1)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_modulator.chunk(
                    6, dim=1
                )
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                        attn_bias=attn_bias,
                        si=si,
                    ).mul_(gamma1)
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(
                    self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(
                        gamma2
                    )
                )
        return x
    
    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"
    
class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        # We achieve multi-token conditioning through LLM attention mask.
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values

        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)

        


        