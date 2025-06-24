import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Regular diffusion imports (not inpainting)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler, 
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

class CustomLoRAGenerator:
    def __init__(self, 
                 model_name="stabilityai/stable-diffusion-2-1",  # Regular SD, not inpainting
                 lora_weights_path="./lora_weights.pth",
                 rank=8,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.rank = rank
        self.lora_weights_path = lora_weights_path
        
        print(f"Using device: {device}")
        print(f"Loading LoRA weights from: {lora_weights_path}")
        
        # Load models
        print("Loading models...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_name, subfolder="feature_extractor")
        
        # Move to device
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        # feature_extractor doesn't need to be moved to device
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Setup LoRA and load custom weights
        self.setup_lora()
        self.load_custom_lora_weights()
        
        # Create pipeline
        self.create_pipeline()
        
        print("Models loaded successfully!")
    
    def setup_lora(self):
        """Add LoRA adapters to UNet"""
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        print("LoRA adapters added to UNet")
    
    def load_custom_lora_weights(self):
        """Load your custom LoRA weights"""
        if not os.path.exists(self.lora_weights_path):
            raise FileNotFoundError(f"LoRA weights file not found: {self.lora_weights_path}")
        
        # Load the state dict
        lora_state_dict = torch.load(self.lora_weights_path, map_location=self.device)
        
        # Set the LoRA weights
        set_peft_model_state_dict(self.unet, lora_state_dict)
        print(f"Custom LoRA weights loaded from {self.lora_weights_path}")
    
    def create_pipeline(self):
        """Create the Stable Diffusion pipeline with custom LoRA"""
        self.pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            feature_extractor=self.feature_extractor,
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False
        )
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_image(self, prompt, negative_prompt="", 
                      num_inference_steps=50, guidance_scale=7.5, 
                      width=512, height=512, seed=None):
        """Generate image using your custom LoRA"""
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
        
        return result.images[0]
    
    def generate_multiple(self, prompts, output_dir="./generated_images", **kwargs):
        """Generate multiple images from a list of prompts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_images = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
            
            try:
                image = self.generate_image(prompt, **kwargs)
                
                # Save image
                output_file = output_path / f"generated_{i+1:03d}.png"
                image.save(output_file)
                
                generated_images.append((prompt, image, output_file))
                print(f"Saved: {output_file}")
                
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                continue
        
        return generated_images

def generate_furniture_images(lora_weights_path, furniture_types=None):
    """Generate images for different furniture types"""
    
    if furniture_types is None:
        furniture_types = [
            "armchair", "beanbag", "bed", "bench", "cabinet", 
            "chair", "coffee table", "sofa", "dining table", "bookshelf"
        ]
    
    # Initialize generator
    generator = CustomLoRAGenerator(
        model_name="stabilityai/stable-diffusion-2-1",
        lora_weights_path=lora_weights_path,
        rank=8
    )
    
    # Create prompts
    prompts = []
    for furniture in furniture_types:
        # You can customize these prompts based on your training data
        base_prompts = [
            f"a photo of a {furniture}",
            f"a modern {furniture}",
            f"a wooden {furniture}",
            f"a comfortable {furniture}",
            f"a stylish {furniture}"
        ]
        prompts.extend(base_prompts)
    
    # Generate images
    generated_images = generator.generate_multiple(
        prompts=prompts,
        output_dir="./generated_furniture",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=50,
        guidance_scale=7.5,
        width=512,
        height=512
    )
    
    return generated_images

def main():
    print("=== Custom LoRA Image Generation ===")
    
    # Option 1: Generate single image
    print("\n1. Single Image Generation:")
    try:
        generator = CustomLoRAGenerator(
            model_name="stabilityai/stable-diffusion-2-1",
            lora_weights_path="./output_model/lora_weights.pth",
            rank=8  # Match your training rank
        )
        
        # Generate a single image
        image = generator.generate_image(
            prompt="a photo of a modern armchair",
            negative_prompt="blurry, low quality",
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42  # For reproducible results
        )
        
        # Save the image
        image.save("./test_generation.png")
        print("Single image saved to: ./test_generation.png")
        
    except Exception as e:
        print(f"Error in single generation: {e}")
    
    # Option 2: Generate multiple furniture images
    print("\n2. Batch Furniture Generation:")
    try:
        furniture_list = ["armchair", "chair", "sofa", "table", "bed"]
        
        generated_images = generate_furniture_images(
            lora_weights_path="./output_model/lora_weights.pth",
            furniture_types=furniture_list
        )
        
        print(f"Generated {len(generated_images)} images successfully!")
        
    except Exception as e:
        print(f"Error in batch generation: {e}")

if __name__ == "__main__":
    main()