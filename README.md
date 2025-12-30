<div align="center">
    <h1> 
        Markovian Scale Prediction: A New Era of Visual Autoregressive Generation
    </h1>
</div>

<div align="center">
    <a href="https://arxiv.org/pdf/2511.23334" style="text-decoration:none; margin: 0 4px;">
        <img src="https://img.shields.io/badge/ArXiv-2511.23334-b31b1b.svg" alt="ArXiv" />
    </a>
    <a href="https://luokairo.github.io/markov-var-page/" style="text-decoration:none; margin: 0 4px;">
        <img src="https://img.shields.io/badge/Github-Project_Page-blue" alt="Project Page" />
    </a>
    <a href="https://huggingface.co/kairoliu/Markov-VAR" style="text-decoration:none; margin: 0 4px;">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Markovar/model-yellow" alt="huggingface weights" />
    </a>
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

Download the repo:

```bash
git clone https://github.com/luokairo/Markov-VAR.git
cd Markov-VAR
conda create -n markov python=3.10
conda activate markov
conda install -c nvidia cuda-toolkit -y
```

### Command Line Inference

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