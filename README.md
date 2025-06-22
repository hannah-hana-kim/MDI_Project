# MDI_Project

## Simple Dreambooth ControlNet Lora Training
### Simplifications Made

#### Removed Dependencies:
- **accelerate**: No distributed training, just single GPU/CPU
- **huggingface_hub**: No automatic downloading, uses local cache
- **wandb/tensorboard**: No fancy logging, just print statements
- **bitsandbytes**: No 8-bit optimization
- **xformers**: No memory optimization

#### Removed Features:
- **Prior preservation**: No class images generation
- **Validation during training**: Just focuses on training
- **Advanced scheduling**: Simple constant learning rate
- **Gradient accumulation**: Direct batch processing
- **Mixed precision**: Uses full precision
- **Checkpointing**: Simple final save only

#### What Remains (Core Logic):
1. **Dataset loading**: Your images → tensors
2. **Random masking**: For inpainting training
3. **LoRA setup**: Adds trainable parameters to UNet
4. **Training loop**: 
   - Encode images to latents
   - Add noise (forward diffusion)
   - Predict noise with UNet
   - Calculate MSE loss
   - Backprop and optimize
5. **Save LoRA weights**: For later use

#### Minimal Requirements:
```
torch>=2.0.0
torchvision
diffusers>=0.20.0
transformers>=4.25.0
peft>=0.5.0
Pillow
numpy
tqdm
```

#### Usage:
```python
# Just run the script - it will:
# 1. Load SD 1.5 from local cache
# 2. Find all images in ./data/
# 3. Train LoRA for 500 steps
# 4. Save weights to ./output/simple_lora/
```