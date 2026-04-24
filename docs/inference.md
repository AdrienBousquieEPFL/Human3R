# Pre-trained Models

We provide multiple checkpoints with different Multi-HMR encoders on [HuggingFace](https://huggingface.co/faneggg/human3r/tree/main), including ViT-672S/B/L and ViT-896L. 
Once downloaded you need to place them into the `src` directory.
```Bash
# Download multiple Human3R checkpoints
hf download faneggg/human3r human3r_672S.pth --local-dir ./src
hf download faneggg/human3r human3r_672B.pth --local-dir ./src
hf download faneggg/human3r human3r_672L.pth --local-dir ./src
hf download faneggg/human3r human3r_896L.pth --local-dir ./src
```


Here is an evaluation of their **Accuracy vs. Speed Trade-off**:

|  | Local Human |  |  | Global Human |  |  | Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Models** | PA-MPJPE↓ | MPJPE↓ | PVE↓ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ | FPS↑ |
| [Human3R w/ ViT-S/672](https://huggingface.co/faneggg/human3r/resolve/main/human3r_672S.pth?download=true) | 56.1 | 87.8 | 103.1 | 129.9 | 314.2 | 2.2 | **15** |
| [Human3R w/ ViT-B/672](https://huggingface.co/faneggg/human3r/resolve/main/human3r_672B.pth?download=true) | 49.3 | 79.6 | 94.3 | 122.1 | 292.9 | 2.2 | 11 |
| [Human3R w/ ViT-L/672](https://huggingface.co/faneggg/human3r/resolve/main/human3r_672L.pth?download=true) | 48.5 | 83.1 | 96.7 | 113.6 | 291.7 | 2.2 | 7 |
| [Human3R w/ ViT-L/896](https://huggingface.co/faneggg/human3r/resolve/main/human3r_896L.pth?download=true) | **44.1** | **71.2** | **84.9** | **112.2** | **267.9** | **2.2** | 5 |

**Benchmark Setup:** All reported speeds are measured on an NVIDIA RTX 4090 GPU with dual Intel Xeon Gold 6530 CPUs.

**Real-Time Tier:** **ViT-S** (15 FPS) offers a strong balance for global motion estimation (WA-MPJPE 129.9, RTE 2.2).  

**High-Fidelity Tier:** **ViT-L** (5-7 FPS) provides more detailed human-mesh reconstruction (WA-MPJPE 112.2, RTE 2.2), suitable for application requiring fine-grained pose and shape.


# Inference Speed

Additionally, we benchmark runtime across diverse datasets and backbones. 
As shown below, the **ViT-S/672** variant indeed supports real-time applications (\~15 FPS), while larger models trade speed for detail:

|  | FPS |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Models** | 3DPW (288×512) | BEDLAM (512×288) | RICH (512×368) | EMDB (384×512) | Bonn (512×384) | TUM-D (512×384) |
| Human3R w/ ViT-S/672 | **15.87** | 15.64 | 14.28 | 13.75 | 13.65 | 13.59 |
| Human3R w/ ViT-B/672 | **13.33** | 12.69 | 11.89 | 11.68 | 11.67 | 12.41 |
| Human3R w/ ViT-L/672 | **9.17** | 8.73 | 8.38 | 8.27 | 8.27 | 8.61 |
| Human3R w/ ViT-L/896 | **5.38** | 5.3 | 5.15 | 5.09 | 5.06 | 5.15 |



We provide a script to evaluate the model inference speed (forward pass only), excluding data loader/saver operations.

> Note: It is recommended to use `inference_only.py` instead of `eval/global_human/launch.py` for timing purposes, as the evaluation script includes significant overhead from ground-truth data processing, metric computation, visualization, and I/O operations.


```bash
    # Example:
    CUDA_VISIBLE_DEVICES=0 python inference_only.py \
        --model_path src/human3r_672S.pth --size 512 \
        --seq_path /path/to/3DPW/imageFiles/downtown_runForBus_00 \
        --use_ttt3r --reset_interval 100
```
