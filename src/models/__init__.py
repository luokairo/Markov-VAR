from typing import Tuple
from requests import head
from sympy import Mod
import torch.nn as nn

from .quant import VectorQuantizer2
from .markovvar import Markov_VAR
from .vqvae import VQVAE


from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32

def Markov_VAR_B(**kwargs):
    config = ModelArgs(n_layer=24, n_head=16, dim=768, **kwargs)
    return config

def Markov_VAR_L(**kwargs):
    config = ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)
    return config

def Markov_VAR_XL(**kwargs):
    config = ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)
    return config

MODEL_REGISTRY = {
    "Markov_VAR_B": Markov_VAR_B,
    "Markov_VAR_L": Markov_VAR_L,
    "Markov_VAR_XL": Markov_VAR_XL,
}

def get_model_config(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name](**kwargs)

def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1, model_name=None    # init_std < 0: automated
) -> Tuple[VQVAE, Markov_VAR]:
    heads = depth
    width = depth * 64
    if model_name is not None:
        config = get_model_config(name=model_name)
        heads = config.n_head
        depth = config.n_layer
        width = config.dim
    else:
        heads = depth
        width = depth * 64

    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    var_wo_ddp = Markov_VAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp
