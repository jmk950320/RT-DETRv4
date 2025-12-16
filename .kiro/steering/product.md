# RT-DETRv4 Product Overview

RT-DETRv4 is a state-of-the-art real-time object detection framework that leverages Vision Foundation Models (VFMs) for enhanced performance. It introduces a cost-effective distillation framework using DINOv3 as a teacher model to improve lightweight detectors.

## Key Features

- **Real-time object detection** with DETR-based architecture
- **Vision Foundation Model distillation** using DINOv3/DINOv2 teachers
- **Multi-model support**: RT-DETRv4, DEIM, D-FINE, RT-DETRv2
- **COCO dataset optimization** with state-of-the-art AP scores
- **Deployment ready** with ONNX and TensorRT export capabilities
- **Multiple backbone options**: HGNetv2, ResNet variants

## Performance Targets

The framework achieves new SOTA results on COCO:
- RT-DETRv4-S: 49.8 AP (273 FPS on T4)
- RT-DETRv4-M: 53.7 AP (169 FPS on T4) 
- RT-DETRv4-L: 55.4 AP (124 FPS on T4)
- RT-DETRv4-X: 57.0 AP (78 FPS on T4)

## Primary Use Cases

- Real-time object detection applications
- Research on DETR-based architectures
- Vision foundation model distillation experiments
- Production deployment with optimized inference