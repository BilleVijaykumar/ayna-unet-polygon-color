# Colored Polygon UNet

## Overview
UNet model trained to generate colored polygon images based on grayscale shapes and color names.

## Architecture
- UNet with color conditioning via channel concatenation

## Training
- Loss: MSE
- Optimizer: Adam
- Epochs: 20
- Tracked with wandb

## Inference
Run `inference.ipynb` to test a sample input.

## Key Learnings
- How to build conditional image generation models
- Integrating structured metadata (text color name) into vision models
- Data pipeline design for multimodal inputs

## To Run
```bash
pip install -r requirements.txt
python train.py
```