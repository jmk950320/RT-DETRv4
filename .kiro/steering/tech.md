# Technology Stack & Build System

## Core Framework
- **PyTorch**: Primary deep learning framework
- **Python 3.11.9**: Recommended Python version
- **CUDA**: GPU acceleration support

## Key Dependencies
- `torch` & `torchvision`: Core PyTorch libraries
- `faster-coco-eval>=1.6.5`: Optimized COCO evaluation
- `PyYAML`: Configuration management
- `tensorboard`: Training visualization
- `scipy`: Scientific computing utilities
- `calflops`: Model complexity analysis
- `transformers`: Hugging Face transformers support

## Development Environment Setup
```bash
conda create -n rtv4 python=3.11.9
conda activate rtv4
pip install -r requirements.txt
```

## Common Commands

### Training
```bash
# Single GPU training
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0


```

### Testing & Evaluation
```bash
# Model evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --test-only -r model.pth

# Fine-tuning from checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0 -t model.pth
```

### Deployment & Export
```bash
# Export to ONNX
python tools/deployment/export_onnx.py --check -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth

# Convert to TensorRT
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

### Inference
```bash
# PyTorch inference
python tools/inference/torch_inf.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth --input image.jpg --device cuda:0

# ONNX inference
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg

# TensorRT inference
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
```

### Benchmarking & Analysis
```bash
# Model complexity analysis
python tools/benchmark/get_info.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml

# TensorRT latency benchmarking
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

## Configuration System
- **YAML-based**: All configurations use YAML format with inheritance via `__include__`
- **Hierarchical**: Base configs in `configs/base/`, model-specific in `configs/rtv4/`
- **CLI overrides**: Use `-u` flag to override config values from command line
- **Distributed training**: Built-in support with `torchrun`

## Teacher Model Requirements
- DINOv3 repository must be available locally
- Pre-trained weights: `dinov3_vitb16_pretrain_lvd1689m.pth`
- Configure paths in model YAML files under `teacher_model` section