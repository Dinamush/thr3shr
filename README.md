---
title: ML-Danbooru ONNX Webapp
emoji: "🖼️"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
tags:
- image-classification
- onnx
- anime
- tagging
- danbooru
- deep-learning
- computer-vision
---

# ML-Danbooru ONNX Models

## Summary

This repository provides **ONNX-optimized** implementations of the **ML-Danbooru** image tagging models, originally developed by 7eu7d7. ML-Danbooru is a sophisticated **deep learning** system specifically designed for **automated tagging** of anime-style images, leveraging modern transformer architectures to achieve high-precision classification across thousands of Danbooru-style tags. The models in this repository have been converted to ONNX format for improved inference performance and cross-platform compatibility.

The core architecture employs **Caformer** (Convolution-Augmented Transformer) models, which combine the global receptive field of transformers with the local feature extraction capabilities of convolutional networks. This hybrid approach enables the models to effectively capture both fine-grained details and global contextual information in anime artwork. The repository includes multiple model variants trained with different configurations and epochs, providing users with options ranging from faster inference to higher accuracy depending on their specific requirements.

Performance-wise, these models demonstrate **exceptional accuracy** in recognizing common anime character attributes, clothing items, accessories, backgrounds, and compositional elements. They can reliably identify tags such as hair colors, eye colors, clothing types, character poses, and scene settings with confidence scores typically exceeding 0.7-0.9 for relevant features. The models support **batch processing** and can handle images of various aspect ratios through intelligent resizing strategies that preserve important visual information while maintaining computational efficiency.

## Usage

The models in this repository are designed to be used with the `dghs-imgutils` library, which provides a comprehensive interface for image tagging tasks.

### Installation

```bash
pip install dghs-imgutils
```

### Basic Usage

```python
from imgutils.tagging import get_mldanbooru_tags

# Tag an image with default settings
tags = get_mldanbooru_tags('your_image.jpg')
print(tags)

# Tag with custom threshold and settings
tags_custom = get_mldanbooru_tags(
    'your_image.jpg',
    threshold=0.5,
    size=448,
    keep_ratio=True,
    drop_overlap=True,
    use_real_name=False
)
print(tags_custom)
```

## Model Variants

This repository contains multiple ML-Danbooru model variants:

- **ml_caformer_m36_dec-5-97527.onnx**: Primary model with Caformer-M36 architecture
- **ml_caformer_m36_dec-3-80000.onnx**: Alternative checkpoint with different training
- **TResnet-D-FLq_ema_2-40000.onnx**: TResnet-based variant
- **TResnet-D-FLq_ema_4-10000.onnx**: Lightweight TResnet variant
- **TResnet-D-FLq_ema_6-10000.onnx**: Additional TResnet checkpoint
- **TResnet-D-FLq_ema_6-30000.onnx**: Extended training TResnet variant
- **caformer_m36-3-80000.onnx**: Base Caformer model

## Tag Information

The repository includes comprehensive tag information:

- **classes.json**: Contains 1,527 simplified tag names for common anime attributes
- **tags.csv**: Complete tag database with 12,547 entries including:
  - Original tag names
  - Root forms for morphological variations
  - Part-of-speech classifications
  - Usage frequency counts

## Performance Characteristics

- **Input Size**: Default 448x448 pixels (configurable)
- **Tag Count**: 12,547 possible tags
- **Threshold**: Default 0.7 (configurable)
- **Supported Tags**: Character attributes, clothing, accessories, backgrounds, compositions
- **Architecture**: Caformer-M36 and TResnet variants
- **Format**: ONNX for optimized inference

### Model Architecture Details

The ML-Danbooru models utilize modern transformer-based architectures:

- **Caformer-M36**: Combines convolutional layers with transformer blocks for efficient feature extraction
- **TResnet-D**: Transformer-enhanced ResNet variants with focal loss optimization
- **ONNX Optimization**: Models are exported with optimized operators for fast inference across different hardware platforms

## Citation

```bibtex
@misc{deepghs_ml_danbooru_onnx,
  title        = {{ML-Danbooru ONNX Models: Optimized Anime Image Tagging}},
  author       = {7eu7d7 and DeepGHS Contributors},
  howpublished = {\url{https://huggingface.co/deepghs/ml-danbooru-onnx}},
  year         = {2023},
  note         = {ONNX-optimized implementations of ML-Danbooru models for efficient anime image tagging with transformer-based architectures},
  abstract     = {This repository provides ONNX-optimized implementations of the ML-Danbooru image tagging models, originally developed by 7eu7d7. ML-Danbooru is a sophisticated deep learning system specifically designed for automated tagging of anime-style images, leveraging modern transformer architectures to achieve high-precision classification across thousands of Danbooru-style tags. The models employ Caformer (Convolution-Augmented Transformer) architectures that combine the global receptive field of transformers with local feature extraction capabilities of convolutional networks, enabling effective capture of both fine-grained details and global contextual information in anime artwork.},
  keywords     = {image-classification, anime, tagging, danbooru, transformer, onnx}
}
```
