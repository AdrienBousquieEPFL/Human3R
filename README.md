<h2 align="center">
  <img src="https://github.com/user-attachments/assets/11fde037-40f6-4406-aa31-b35ae7f2d381" height="24" style="vertical-align: bottom; margin-right: 0px;" />
  <a href="https://fanegg.github.io/Human3R">Human3R: Everyone Everywhere All at Once</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.06219-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.06219) 
[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://fanegg.github.io/Human3R) 
[![X](https://img.shields.io/badge/@Yue%20Chen-black?logo=X)](https://twitter.com/faneggchen)  [![Bluesky](https://img.shields.io/badge/@Yue%20Chen-white?logo=Bluesky)](https://bsky.app/profile/fanegg.bsky.social)


[Yue Chen](https://fanegg.github.io/),
[Xingyu Chen](https://rover-xingyu.github.io/)*,
[Yuxuan Xue](https://yuxuan-xue.com/),
[Anpei Chen](https://apchenstu.github.io/),
[Yuliang Xiu](https://xiuyuliang.cn/)†,
[Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/)
</h5>

<div align="center">
TL;DR: Inference with One model, One stage; Training in One day using One GPU
</div>
<br>

https://github.com/user-attachments/assets/47fc7ecf-5235-471c-84b9-ccfeca6d56ea

## Getting Started

### Installation

1. Clone Human3R.
```bash
git clone https://github.com/fanegg/Human3R.git
cd Human3R
```

2. Create the environment.
```bash
conda create -n human3r python=3.11 cmake
conda activate human3r

# PyTorch — pick the build that matches your GPU.
# Most setups (CUDA 12.4):
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
# Blackwell GPUs (RTX 5060/5070/5080/5090) need CUDA 12.8 + nightly torch instead:
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# compilers + ninja for building the cuda kernels below (gsplat, pytorch3d, RoPE)
conda install -y gcc_linux-64 gxx_linux-64
pip install ninja

pip install git+https://github.com/nerfstudio-project/gsplat.git

# VolumetricSMPL + pytorch3d (used by demo.py for the volumetric body model / penetration eval).
# pytorch3d has no prebuilt wheels for recent torch/CUDA combos, so build from source:
pip install fvcore iopath
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@main"
pip install VolumetricSMPL

# for evaluation
pip install evo
pip install open3d
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download
Run the following commands to download all models and checkpoints into the `src/` directory. The first command will prompt you to register and log in to access each version of SMPL.
```Bash
# SMPLX family models
bash scripts/fetch_smplx.sh

# Human3R checkpoints
huggingface-cli download faneggg/human3r human3r_896L.pth --local-dir ./src
```

### Inference Demo

To run the inference demo, you can use the following command:
```bash
# input can be a folder or a video
# the following script will run inference with Human3R and visualize the output with viser on port 8080
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path MODEL_PATH --size 512 \
    --seq_path SEQ_PATH --output_dir OUT_DIR --subsample 1 --use_ttt3r \
    --vis_threshold 2 --downsample_factor 1 --reset_interval 100

# Example:
# To save the results, append `--save --output_dir tmp` to the command.
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/human3r_896L.pth \
    --size 512 --seq_path examples/GoodMornin1.mp4 \
    --subsample 1 --use_ttt3r --vis_threshold 2 \
    --downsample_factor 1 --reset_interval 100

```

### Evaluation
Please refer to the [eval.md](docs/eval.md) for more details.

### Model Cards
Please refer to the [inference.md](docs/inference.md) for using different backbones.

### Training
Please refer to the [train.md](docs/train.md) for more details.

## Acknowledgements
Our code is based on the following awesome repositories:

- [CUT3R](https://github.com/CUT3R/CUT3R), [TTT3R](https://github.com/Inception3D/TTT3R), [Multi-HMR](https://github.com/naver/multi-hmr), [PromptHMR](https://github.com/yufu-wang/PromptHMR), [GVHMR](https://github.com/zju3dv/GVHMR), [MonST3R](https://github.com/Junyi42/monst3r.git), [Easi3R](https://github.com/Inception3D/Easi3R), [DUSt3R](https://github.com/naver/dust3r), [Viser](https://github.com/nerfstudio-project/viser), [BEDLAM](https://github.com/pixelite1201/BEDLAM)

We thank the authors for releasing their code!

## Citation

If you find our work useful, please cite:

```bibtex
@article{chen2025human3r,
    title={Human3R: Everyone Everywhere All at Once},
    author={Chen, Yue and Chen, Xingyu and Xue, Yuxuan and Chen, Anpei and Xiu, Yuliang and Gerard, Pons-Moll},
    journal={arXiv preprint arXiv:2510.06219},
    year={2025}
    }
```
