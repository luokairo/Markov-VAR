import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
from glob import glob
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from PIL import Image
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import sys
from models import VQVAE, build_vae_var
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

MODEL_DEPTH = 24
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

vae_ckpt = ''
var_ckpt = ''


patch_nums = (1,2,3,4,5,6,8,10,13,16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var_ckpt = torch.load(var_ckpt, map_location='cpu')
var_wo_ddp_state = var_ckpt['trainer']['var_wo_ddp']
var.load_state_dict(var_wo_ddp_state, strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'var prepare finished.')

def stat(name, t):
    t = t.detach().float().cpu()
    print(name, 'min/max/mean =', t.min().item(), t.max().item(), t.mean().item())

# === CHANGED === 更稳的张量→PNG：自动适配 [-1,1]/[0,1]，显式 PNG
def chw_float_to_pil(x_3hw: torch.Tensor):
    x = x_3hw.detach().cpu().float()
    if x.min() < -0.01 or x.max() > 1.01:
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    return to_pil_image(x)

seed = 0
torch.manual_seed(seed)
num_sampling_steps = 250
cfg = 5  
num_classes = 1000
num_per_class = 50
class_labels = torch.arange(1000)

more_smooth = False
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

USE_FP16_AUTOMATIC_CAST = True  
TARGET_PER_CLASS = 50

base_dir = f""
var_out_root = osp.join(base_dir, 'cfg-5')

B = 50

for cls in tqdm(class_labels):
    cls_idx = int(cls.item())
    var_class_dir = osp.join(var_out_root, f'{cls_idx:03d}')
    os.makedirs(var_class_dir, exist_ok=True)
    
    done = len(glob(osp.join(var_class_dir, "*.png")))
    if done >= TARGET_PER_CLASS:
        print(f"this {var_out_root}class-({cls}) is full")
        if done > TARGET_PER_CLASS:
            print(done)
        continue
    for i in range(50 // B):
        label_B = torch.tensor([cls] * B, device=device)
        # VAR
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):  # using bfloat16 can be faster
                img_var = var.markov_cache_infer_cfg(
                    B,
                    label_B=label_B,
                    cfg=cfg,
                    top_k=900,
                    top_p=0.95,
                    more_smooth=more_smooth,
                    g_seed=int(seed + cls * (50 // B) + i),
                )
            img_var = img_var.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
        img_var = img_var.astype(np.uint8)
        for j in range(B):
            img = PImage.fromarray(img_var[j])
            img.save(f"{var_class_dir}/{(cls * 50 + i * B + j):06d}.png")
            if j == 24:
                print(f"save in {var_class_dir}")