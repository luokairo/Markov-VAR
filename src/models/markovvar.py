import math
import os
from functools import partial
from re import L
from turtle import forward
from typing import Optional, Tuple, Union
from typing import List
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import dist

from models.basic_markovvar import(
    AdaLNBeforeHead,
    LlamaAdaLNSelfAttn,
    RMSNorm,
    TimestepEmbedder,
    ContextAttentivePool
)
from models.utils import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.data
        return super().forward(cond_BD).view(-1, 1, 6, C)
def get_edit_mask(patch_nums: List[int], y0: float, x0: float, y1: float, x1: float, device, inpainting: bool = True) -> torch.Tensor:
    ph, pw = patch_nums[-1], patch_nums[-1]
    edit_mask = torch.zeros(ph, pw, device=device)
    edit_mask[round(y0 * ph):round(y1 * ph), round(x0 * pw):round(x1 * pw)] = 1 # outpainting mode: center would be gt
    if inpainting:
        edit_mask = 1 - edit_mask   # inpainting mode: center would be model pred
    return edit_mask

def replace_embedding(edit_mask: torch.Tensor, h_BChw: torch.Tensor, gt_BChw: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    B = h_BChw.shape[0]
    h, w = edit_mask.shape[-2:]
    if edit_mask.ndim == 2:
        edit_mask = edit_mask.unsqueeze(0).expand(B, h, w)

    force_gt_B1hw = F.interpolate(edit_mask.unsqueeze(1).to(dtype=torch.float, device=gt_BChw.device), size=(ph, pw), mode='bilinear', align_corners=False).gt(0.5).int()
    if ph * pw <= 3: force_gt_B1hw.fill_(1)
    return gt_BChw * force_gt_B1hw + h_BChw * (1 - force_gt_B1hw)


class Markov_VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, disable_aln=False,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []

        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        norm_layer = partial(RMSNorm, eps=norm_eps)
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. rope
        self.pos_start = None
        self.pos_1LC = None
        self.lvl_embed = TimestepEmbedder(embed_dim)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Identity()
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            LlamaAdaLNSelfAttn(
                block_idx=block_idx,
                norm_layer=norm_layer,
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                embed_dim=self.C,
                cond_dim=self.D,
                shared_aln=shared_aln,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,
                fused_if_available=fused_if_available,
                disable_aln=disable_aln,
                max_position_embeddings=2 ** int(math.ceil(math.log2(self.L))),
                patch_nums=patch_nums,
                context_token=1,
                rope_theta=10000,
                use_cross_attn=False
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [MNTP config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training
        d: torch.Tensor = torch.cat([
            torch.full((pn * pn,), i, dtype=torch.long) for i, pn in enumerate(self.patch_nums)
        ]).view(1, self.L, 1)

        # 转置得到 dT
        dT = d.transpose(1, 2)  # shape (1, 1, L)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        allow = (d == dT)

        attn_bias_for_masking = torch.where(allow, 0., -torch.inf).reshape(1, 1, self.L, self.L)

        # 注册 buffer
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        print(attn_bias_for_masking.shape)

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # 7. window
        self.context_pool = ContextAttentivePool(self.C, self.D)
        self.context_cache = []
        self.head_context = nn.Linear(2 * self.D, self.D)
        self.norm_x = nn.LayerNorm(self.D)
        self.norm_cond = nn.LayerNorm(self.C)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def get_context_cache_wo_first_l(self, x_BLC_wo_first_l, window_size=3):
        context_list = []

        for (begin, end) in self.begin_ends[1:]:
            begin -= self.first_l
            end -= self.first_l
            x_slice = x_BLC_wo_first_l[:, begin:end, :]
            context_list.append(x_slice)
        
        context_pooled_list = []
        num_scales = len(context_list)
        for i in range(num_scales):
            start_idx = max(0, i - window_size + 1)
            window_slices = context_list[start_idx : i + 1]
            cat_slice = torch.cat(window_slices, dim=1)
            pooled = self.context_pool(cat_slice)
            context_pooled_list.append(pooled)
        
        return context_pooled_list
    
    def fuse_context_and_cond_BD_wo_first_l(self, context_cache_list, cond_BD):
        """
        cond_BD: (B, D)
        context_cache_list: List[(B, D)]
        """
        B, D = cond_BD.shape
        
        cond_BLD = cond_BD.new_zeros(B, self.L - self.first_l, D)
        for i, (be, ed) in enumerate(self.begin_ends[1:]):
            be -= self.first_l
            ed -= self.first_l
            cond_context = context_cache_list[i].unsqueeze(1).expand(B, ed - be, D)
            cond_BLD[:, be:ed, :] = cond_context + cond_BD.unsqueeze(1)

        return cond_BLD
    
    def get_context_cache_infer(self, context_cache_list, cond_BD, cur_cond_L, window_size=3):
        """
        context_cache: List[(B, L_si, C)]
        cond_BD: (B, D)

        (1, 2, 3)
        (14)
        ([0:1], [1:5], [5:14])

        return: cond_BLD : (B, L, D)
        x_BLD
        """
        cur_len = len(context_cache_list)
        k = min(window_size, cur_len)
        context_pooled_list = context_cache_list[-k:]
        context_pooled = torch.cat(context_pooled_list, dim=1)
        context_pooled = self.context_pool(context_pooled)

        B2, D = cond_BD.shape
        cond_context = context_pooled.unsqueeze(1).expand(B2, cur_cond_L, D)
        cond_BLD = cond_context + cond_BD.unsqueeze(1)

        return cond_BLD
    
    @torch.no_grad()
    def markov_cache_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ):
        self.context_cache.clear()
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        total_kv_cache_mem = 0
        total_intermediate_mem = 0
        max_memory_usage = 0
        
        for b in self.blocks:
            b.attn.kv_caching(False)

        # 在循环前记录基础显存(模型权重)
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        # print(f"Base memory (model weights): {base_mem:.2f} MB\n")
        
        for si, pn in enumerate(self.patch_nums):
            stage_start_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cur_cond_L = pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            x = next_token_map
            LlamaAdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                )
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map)
                next_token_map = self.norm_x(next_token_map)

                # 判断如果 context_cache 中的序列数达到 3，进行出队操作
                if len(self.context_cache) == 3:
                    self.context_cache.pop(0)  # 移除最旧的元素

                # 添加新的元素到 context_cache
                self.context_cache.append(next_token_map)
                cur_cond_L = next_token_map.shape[1]
                cond_BLD = self.get_context_cache_infer(self.context_cache, cond_BD[:B], cur_cond_L)
                cond_BLD = self.norm_cond(cond_BLD)

                next_token_map = torch.cat((next_token_map, cond_BLD), dim=-1)
                next_token_map = self.head_context(next_token_map)

                next_token_map += lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

            # 记录 stage 结束时的显存
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            stage_increment = current_mem - stage_start_mem
            
            # print(f"Stage {si}:")
            # print(f"  - Cumulative peak: {peak_mem:.2f} MB")
            # print(f"  - Current allocated: {current_mem:.2f} MB")
            # print(f"  - Stage increment: {stage_increment:.2f} MB")

            torch.cuda.empty_cache()
        for b in self.blocks: b.attn.kv_caching(False)
        result = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        # 记录 stage 结束时的显存
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        stage_increment = current_mem - stage_start_mem
        # print(f"decoder")
        # print(f"  - Cumulative peak: {peak_mem:.2f} MB")
        # print(f"  - Current allocated: {current_mem:.2f} MB")
        # print(f"  - Stage increment: {stage_increment:.2f} MB")
        return result

    def markov_infer_cfg_with_mask(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
        input_img_tokens: Optional[List[torch.Tensor]] = None, edit_mask: Optional[torch.Tensor] = None,
    ):
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks:
            b.attn.kv_caching(False)
        
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cur_cond_L = pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            x = next_token_map
            LlamaAdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            if edit_mask is not None:
                gt_BChw = self.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                h_BChw = replace_embedding(edit_mask, h_BChw, gt_BChw, pn, pn)
            
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map)
                next_token_map = self.norm_x(next_token_map)

                # 判断如果 context_cache 中的序列数达到 3，进行出队操作
                if len(self.context_cache) == 3:
                    self.context_cache.pop(0)  # 移除最旧的元素

                # 添加新的元素到 context_cache
                self.context_cache.append(next_token_map)
                cur_cond_L = next_token_map.shape[1]
                cond_BLD = self.get_context_cache_infer(self.context_cache, cond_BD[:B], cur_cond_L)
                cond_BLD = self.norm_cond(cond_BLD)

                next_token_map = torch.cat((next_token_map, cond_BLD), dim=-1)
                next_token_map = self.head_context(next_token_map)

                next_token_map += lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        for b in self.blocks: b.attn.kv_caching(False)
        result = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        return result

    @torch.no_grad()
    def markov_cache_infer_cfg_turb(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
        disturb_si = 1,           # 要加扰动的尺度编号
        disturb_idx = 10,         # token 位置索引
        disturb_strength = 2.0,   # 扰动强度
        disturb_type = "logits"  # 控制扰动类型
    ):
        self.context_cache.clear()
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks:
            b.attn.kv_caching(False)
        
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cur_cond_L = pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            x = next_token_map
            LlamaAdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            # # ----------- 加扰动点 -----------
            # if si == disturb_si and disturb_type == "logits":
            #     disturb_idx = round(pn*pn/2)
            #     print(f"Applying disturbance on logits at si={si}, idx={disturb_idx}")
            #     logits_BlV[:, disturb_idx, :] += disturb_strength * torch.randn_like(logits_BlV[:, disturb_idx, :])
            # # --------------------------------
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]

            # ----------- 加扰动点 -----------
            if si == disturb_si:
                disturb_pos = torch.randint(0, idx_Bl.size(1), (B,)).to(idx_Bl.device)  # 随机选择要扰动的token位置
                disturb_value = torch.randint(0, logits_BlV.size(-1), (B,)).to(idx_Bl.device)  # 选择新的token值
                idx_Bl[torch.arange(B), disturb_pos] = disturb_value  # 对选择的token位置应用扰动
            # ----------- 加扰动点 -----------

            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map)
                next_token_map = self.norm_x(next_token_map)

                self.context_cache.append(next_token_map)
                cur_cond_L = next_token_map.shape[1]
                cond_BLD = self.get_context_cache_infer(self.context_cache, cond_BD[:B], cur_cond_L)
                cond_BLD = self.norm_cond(cond_BLD)

                next_token_map = torch.cat((next_token_map, cond_BLD), dim=-1)
                next_token_map = self.head_context(next_token_map)

                next_token_map += lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor):
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1)
            
            x_BLC_wo_first_l = self.word_embed(x_BLCv_wo_first_l.float())
            x_BLC_wo_first_l = self.norm_x(x_BLC_wo_first_l)
            context_cache_list = self.get_context_cache_wo_first_l(x_BLC_wo_first_l)
            cond_BLD_wo_first_l = self.fuse_context_and_cond_BD_wo_first_l(context_cache_list, cond_BD)
            cond_BLD_wo_first_l = self.norm_cond(cond_BLD_wo_first_l)

            x_BLC_wo_first_l = torch.cat((x_BLC_wo_first_l, cond_BLD_wo_first_l), dim=-1)
            x_BLC_wo_first_l = self.head_context(x_BLC_wo_first_l)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, x_BLC_wo_first_l.float()), dim=1)

            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

         # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        LlamaAdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(
                x=x_BLC,
                cond_BD=cond_BD_or_gss,
                attn_bias=attn_bias,
            ) 
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        return x_BLC 
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: LlamaAdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.down_proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                if isinstance(sab.ada_lin, nn.Sequential):
                    target_lin = sab.ada_lin[-1]  
                elif hasattr(sab.ada_lin, 'fc') and isinstance(sab.ada_lin.fc, nn.Linear):
                    target_lin = sab.ada_lin.fc 
                else:
                    target_lin = None

                if target_lin is not None:
                    target_lin.weight.data[2*self.C:].mul_(init_adaln)
                    target_lin.weight.data[:2*self.C].mul_(init_adaln_gamma)
                    if getattr(target_lin, 'bias', None) is not None:
                        target_lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

