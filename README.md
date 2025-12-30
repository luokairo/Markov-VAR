<div align="center">
    <h1> 
        Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
    </h1>
</div>

<div align="center">
    
[![arxiv](https://img.shields.io/badge/ArXiv-2511.23334-b31b1b.svg)](https://arxiv.org/pdf/2511.23334)&nbsp;
[![Project Page](https://img.shields.io/badge/Github-Project_Page-blue)](https://luokairo.github.io/markov-var-page/)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Markovar/model-yellow)](https://huggingface.co/kairoliu/Markov-VAR)&nbsp;

</div>

<div align="center">
<img alt="image" src="assets/framework.jpg" style="width:98%;">
</div>

<div align="center">
<strong>Figure 1: Markov-VAR Framework Overview</strong>
</div>

## News

- [2025-12] We release the pretrained **Markov-VAR** weights on 
  [Hugging Face](https://huggingface.co/kairoliu/Markov-VAR).
- [2025-11] We open-source the code for **Markov-VAR**.
- [2025-11] The paper is now available on arXiv!

## Abstract

Visual AutoRegressive modeling (VAR) based on next-scale prediction has revitalized autoregressive visual generation. Although its full-context dependency, i.e., modeling all previous scales for next-scale prediction, facilitates more stable and comprehensive representation learning by leveraging complete information flow, the resulting computational inefficiency and substantial overhead severely hinder VAR's practicality and scalability. This motivates us to develop a new VAR model with better performance and efficiency without full-context dependency. To address this, we reformulate VAR as a non-full-context Markov process, proposing Markov-VAR. It is achieved via Markovian Scale Prediction: we treat each scale as a Markov state and introduce a sliding window that compresses certain previous scales into a compact history vector to compensate for historical information loss owing to non-full-context dependency. Integrating the history vector with the Markov state yields a representative dynamic state that evolves under a Markov process.  Extensive experiments demonstrate that Markov-VAR is extremely simple yet highly effective: Compared to VAR on ImageNet, Markov-VAR reduces FID by 10.5\% (256×256) and decreases peak memory consumption by 83.8\% (1024×1024). We believe that Markov-VAR can serve as a foundation for future research on visual autoregressive generation and other downstream tasks.

## Installation

### Download the repo

```bash
git clone https://github.com/luokairo/Markov-VAR.git
cd Markov-VAR
```

### Install requirements

```
conda create -n markov python=3.10
conda activate markov
conda install -c nvidia cuda-toolkit -y
pip install -r requirements.txt
```

### Download tokenizer and models

```
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth

git clone https://huggingface.co/kairoliu/Markov-VAR
```

### Python Inference Example

```
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
from glob import glob
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from PIL import Image
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  

from models import VQVAE, build_vae_markovvar
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

MODEL_DEPTH = 24
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

vae_ckpt = 'path/vae_ckpt'
markovvar_ckpt = f'path/Markov-VAR/d{MODEL_DEPTH}.pth'

patch_nums = (1,2,3,4,5,6,8,10,13,16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'vae' not in globals() or 'var' not in globals():
    vae, markovvar = build_vae_markovvar(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
markovvar_ckpt = torch.load(markovvar_ckpt, map_location='cpu')
var_wo_ddp_state = markovvar_ckpt['trainer']['var_wo_ddp']
markovvar.load_state_dict(var_wo_ddp_state, strict=True)
vae.eval(), markovvar.eval()

print(f'var prepare finished.')

seed = 0
torch.manual_seed(seed)
num_sampling_steps = 250
cfg = 5  
num_classes = 1000
num_per_class = 50
class_labels = torch.randint(0, 1000, (10,))

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
B = 50

out_root = "cfg-5"

for cls in class_labels[:10]:  # example: generate for 10 classes
    cls_idx = int(cls)
    out_dir = osp.join(out_root, f"{cls_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)

    label = torch.full((B,), cls_idx, device=device)

    with torch.inference_mode(), torch.autocast(
        "cuda", enabled=USE_FP16_AUTOMATIC_CAST, dtype=torch.float16
    ):
        imgs = markovvar.markov_cache_infer_cfg(
            B,
            label_B=label,
            cfg=cfg,
            top_k=900,
            top_p=0.95,
            g_seed=cls_idx,
        )

    imgs = (
        imgs.permute(0, 2, 3, 1)
        .mul(255)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )

    for i in range(B):
        PImage.fromarray(imgs[i]).save(
            osp.join(out_dir, f"{i:06d}.png")
        )

```



### Training Scripts

 <p align="center">:construction: :pick: :hammer_and_wrench: :construction_worker:</p>
 <p align="center">Under construction. </p>

## Citation

```bibtex
@article{zhang2025markovian,
  title={Markovian Scale Prediction: A New Era of Visual Autoregressive Generation},
  author={Zhang, Yu and Liu, Jingyi and Shi, Yiwei and Zhang, Qi and Miao, Duoqian and Wang, Changwei and Cao, Longbing},
  journal={arXiv preprint arXiv:2511.23334},
  year={2025}
}
```

---
:star: If Markov-VAR is helpful to your projects, please help star this repo. Thanks! :hugs: