# LHM++ Installation

## Requirements

- Linux (tested on Ubuntu)
- Python 3.10
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 12.1 (recommended)

## 1. Clone the repository

```bash
git clone https://github.com/aigc3d/LHM-plusplus
cd LHM-plusplus
```

## 2. Install PyTorch and xformers

```bash
# CUDA 12.1 (recommended)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install base dependencies

```bash
pip install -r requirements.txt
pip install rembg[cpu]  # used during extracting sparse view inputs
```

## 4. Install pointops

```bash
cd ./lib/pointops/ && python setup.py install && cd ../../
```

## 5. Install spconv and torch_scatter

```bash
pip install spconv-cu121

# torch_scatter: see [wheel](https://data.pyg.org/whl/) for your CUDA version
# Example (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install torch_scatter-2.1.2+pt23cu121-cp310-cp310-linux_x86_64.whl
```

## 6. Install PyTorch3D

```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html
```

## 7. Install diff-gaussian-rasterization

```bash
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
# or
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization
```

## 8. Install simple-knn

```bash
pip install git+https://github.com/camenduru/simple-knn/
```

## 9. Install gsplat

Download the pre-compiled wheel from [gsplat whl](https://docs.gsplat.studio/whl/gsplat/).

```bash
# Example (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
```

## 10. Download model weights

```bash
# Download prior models + pretrained weights (default)
python scripts/download_pretrained_models.py

# Prior models only (human_model_files, voxel_grid, BiRefNet, etc.)
python scripts/download_pretrained_models.py --prior

# LHM++ model weights only (LHMPP-700M, LHMPP-700MC, LHMPPS-700M)
python scripts/download_pretrained_models.py --models
```

## Optional dependencies

### SAM2 (for video segmentation)

We use a modified version of SAM2. Install only if needed for video processing:

```bash
pip install git+https://github.com/hitsz-zuoqi/sam2/
# or
# git clone --recursive https://github.com/hitsz-zuoqi/sam2
# pip install ./sam2
```

## Windows installation

1. Install **Python 3.10** from [python.org](https://www.python.org/downloads/release/python-3100/).
2. Install CUDA 12.1 toolkit.
3. Create a virtual environment and follow steps 2–10 above:

```bash
python -m venv lhmpp_env
lhmpp_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ... then run the remaining pip install commands from steps 3–9
```

Note: Adjust wheel filenames (torch_scatter, gsplat) for your Python version and CUDA. See [PyG wheels](https://data.pyg.org/whl/) and [gsplat whl](https://docs.gsplat.studio/whl/gsplat/).

---

The installation has been tested with Python 3.10 and CUDA 12.1. For issues, refer to [README.md](README.md) or open an issue on GitHub.
