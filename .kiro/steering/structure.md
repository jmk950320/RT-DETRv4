# Project Structure & Organization

## Root Directory Layout

```
RT-DETRv4/
├── configs/           # Configuration files (YAML-based)
├── dinov3/           # DINOv3 teacher model submodule
├── engine/           # Core framework implementation
├── figures/          # Documentation images
├── logs/             # Training logs
├── pretrain/         # Pre-trained model weights
├── tools/            # Utilities and scripts
├── train.py          # Main training entry point
└── requirements.txt  # Python dependencies
```

## Configuration Architecture (`configs/`)

### Hierarchical Structure
- `base/`: Base configuration templates
  - `dataloader.yml`: Data loading configurations
  - `optimizer.yml`: Optimization settings
  - `rtv4.yml`: RT-DETRv4 specific base config
- `dataset/`: Dataset-specific configurations
  - `coco_detection.yml`: COCO dataset setup
  - `custom_detection.yml`: Custom dataset template
- `rtv4/`: Model variant configurations
  - `rtv4_hgnetv2_{s,m,l,x}_coco.yml`: Size variants
- `deim/`, `dfine/`, `rtv2/`: Other supported models

### Configuration Inheritance
- Uses `__include__` directive for composition
- Base configs provide defaults, specific configs override
- Example: `rtv4_hgnetv2_s_coco.yml` includes `dfine_hgnetv2_s_coco.yml` and `rtv4.yml`

## Engine Framework (`engine/`)

### Core Components
- `core/`: Configuration management and workspace utilities
  - `yaml_config.py`: YAML configuration parser with property-based lazy loading
  - `workspace.py`: Object creation and dependency injection
- `solver/`: Training and evaluation orchestration
  - `det_solver.py`: Detection task solver
  - `_solver.py`: Base solver with training loop
- `data/`: Data loading and preprocessing
  - `dataset/`: Dataset implementations (COCO, VOC)
  - `transforms/`: Data augmentation pipeline
- `backbone/`: Neural network backbones
  - `hgnetv2.py`: HGNetv2 implementation
  - `csp_resnet.py`: CSP-ResNet variants
- `rtv4/`: RT-DETRv4 specific components
  - `rtv4.py`: Main model architecture
  - `dinov3_teacher.py`: Teacher model integration
  - `rtv4_criterion.py`: Loss functions with distillation

### Architecture Patterns
- **Registry Pattern**: Components registered via decorators for YAML instantiation
- **Factory Pattern**: `workspace.create()` function for object creation
- **Property-based Lazy Loading**: YAMLConfig loads components on-demand
- **Distributed Training**: Built-in multi-GPU support via `dist_utils`

## Tools Directory (`tools/`)

### Organized by Function
- `deployment/`: Model export utilities
  - `export_onnx.py`: ONNX conversion
  - `export_yolo_w_nms.py`: YOLO format export
- `inference/`: Inference scripts for different backends
  - `torch_inf.py`: PyTorch inference
  - `onnx_inf.py`: ONNX Runtime inference
  - `trt_inf.py`: TensorRT inference
- `benchmark/`: Performance analysis tools
- `dataset/`: Dataset preprocessing utilities
- `visualization/`: Visualization tools (FiftyOne integration)

## Code Organization Principles

### Module Structure
- Each major component has its own directory with `__init__.py`
- Related functionality grouped together (e.g., all transforms in `transforms/`)
- Clear separation between core framework and model-specific code

### Naming Conventions
- **Files**: Snake_case (e.g., `yaml_config.py`)
- **Classes**: PascalCase (e.g., `YAMLConfig`, `DetSolver`)
- **Functions/Variables**: Snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Config files**: Descriptive names with model/dataset info

### Import Patterns
- Relative imports within packages
- Absolute imports from root for cross-package dependencies
- Registry imports in `__init__.py` files for automatic registration

### File Responsibilities
- `train.py`: CLI entry point, argument parsing, solver orchestration
- `engine/solver/`: Training loops, validation, checkpointing
- `engine/core/`: Configuration management, object creation
- `engine/rtv4/`: Model architecture, loss functions, teacher models
- `configs/`: Declarative configuration, no code logic

## Development Workflow

### Adding New Models
1. Implement in `engine/rtv4/` or appropriate module
2. Register components via decorators
3. Create configuration in `configs/`
4. Add solver support if needed

### Adding New Datasets
1. Implement dataset class in `engine/data/dataset/`
2. Create configuration in `configs/dataset/`
3. Update evaluator if custom metrics needed

### Configuration Best Practices
- Use base configs for shared settings
- Override only necessary parameters in specific configs
- Document configuration options in comments
- Test configuration inheritance chains