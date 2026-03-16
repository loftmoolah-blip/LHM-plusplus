# LHM++ 安装说明

## 环境要求

- Linux（已在 Ubuntu 下测试）
- Python 3.10
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 12.1（推荐）

## 1. 克隆仓库

```bash
git clone https://github.com/aigc3d/LHM-plusplus
cd LHM-plusplus
```

## 2. 安装 PyTorch 与 xformers

```bash
# CUDA 12.1（推荐）
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. 安装基础依赖

```bash
pip install -r requirements.txt
pip install rembg[cpu]  # 用于提取稀疏视角输入
```

## 4. 安装 pointops

```bash
cd ./lib/pointops/ && python setup.py install && cd ../../
```

## 5. 安装 spconv 与 torch_scatter

```bash
pip install spconv-cu121

# torch_scatter：请根据 CUDA 版本选择 [wheel](https://data.pyg.org/whl/)
# 示例 (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install torch_scatter-2.1.2+pt23cu121-cp310-cp310-linux_x86_64.whl
```

## 6. 安装 PyTorch3D

```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html
```

## 7. 安装 diff-gaussian-rasterization

```bash
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
# 或
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization
```

## 8. 安装 simple-knn

```bash
pip install git+https://github.com/camenduru/simple-knn/
```

## 9. 安装 gsplat

从 [gsplat whl](https://docs.gsplat.studio/whl/gsplat/) 下载预编译 wheel。

```bash
# 示例 (PyTorch 2.3 + CUDA 12.1 + Python 3.10):
pip install gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl
```

## 10. 下载模型权重

```bash
# 下载先验模型 + 预训练权重（默认）
python scripts/download_pretrained_models.py

# 仅先验模型 (human_model_files, voxel_grid, BiRefNet 等)
python scripts/download_pretrained_models.py --prior

# 仅 LHM++ 模型权重 (LHMPP-700M, LHMPP-700MC, LHMPPS-700M)
python scripts/download_pretrained_models.py --models
```

## 可选依赖

### SAM2（用于视频分割）

我们使用修改版 SAM2，仅在需要处理视频时安装：

```bash
pip install git+https://github.com/hitsz-zuoqi/sam2/
# 或
# git clone --recursive https://github.com/hitsz-zuoqi/sam2
# pip install ./sam2
```

## Windows 安装

1. 从 [python.org](https://www.python.org/downloads/release/python-3100/) 安装 **Python 3.10**。
2. 安装 CUDA 12.1 工具包。
3. 创建虚拟环境并执行上述步骤 2–10：

```bash
python -m venv lhmpp_env
lhmpp_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ... 然后依次执行步骤 3–9 中的 pip install 命令
```

注意：请根据你的 Python 版本和 CUDA 版本调整 wheel 文件名（torch_scatter、gsplat）。参见 [PyG wheels](https://data.pyg.org/whl/) 和 [gsplat whl](https://docs.gsplat.studio/whl/gsplat/)。

---

安装已在 Python 3.10 和 CUDA 12.1 环境下测试。如有问题，请参考 [README_CN.md](README_CN.md) 或于 GitHub 提交 issue。
