import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random
import math
from tqdm import tqdm

# core diffusion imports
from diffusers import (
    AutoencoderKL,
    DDPMScheduler, 
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

class SimpleDreamBoothDataset(Dataset):
    def __init__(self, data_dir, prompt, resolution=512):
        self.data_dir = Path(data_dir)
        self.prompt = prompt
        self.resolution = resolution
        
        # get all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
        
        print(f"Found {len(self.image_paths)} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and process image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution))
        
        # Convert to tensor and normalize to [-1, 1]
        image_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Create random mask for inpainting
        mask = self.create_random_mask()
        masked_image = image_tensor * (mask < 0.5)
        
        return {
            'pixel_values': image_tensor,
            'masked_image': masked_image,
            'mask': mask,
            'prompt': self.prompt
        }
    
    def create_random_mask(self):
        """Create random rectangular mask"""
        mask = torch.zeros(1, self.resolution, self.resolution)
        
        # Random mask size (10-50% of image)
        mask_size_ratio = random.uniform(0.1, 0.5)
        mask_h = int(self.resolution * mask_size_ratio)
        mask_w = int(self.resolution * mask_size_ratio)
        
        # Random position
        start_h = random.randint(0, self.resolution - mask_h)
        start_w = random.randint(0, self.resolution - mask_w)
        
        mask[:, start_h:start_h+mask_h, start_w:start_w+mask_w] = 1.0
        return mask

def simple_tokenize(tokenizer, prompt):
    """Simple tokenization without padding complications"""
    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    return tokens.input_ids

class SimpleDreamBoothTrainer:
    def __init__(self, 
                 model_name="stabilityai/stable-diffusion-2-inpainting",  # Use inpainting model
                 rank=8,
                 learning_rate=1e-4,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.rank = rank
        self.learning_rate = learning_rate
        
        print(f"Using device: {device}")
        
        # Load models locally (no HF hub needed if already cached)
        print("Loading models...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # Move to device
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Add LoRA to UNet
        self.setup_lora()
        
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
        self.unet.print_trainable_parameters()
    
    def train(self, data_dir, prompt, num_epochs=1, batch_size=1, max_steps=500):
        """Main training loop"""
        
        # Create dataset and dataloader
        dataset = SimpleDreamBoothDataset(data_dir, prompt)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        
        # Training loop
        self.unet.train()
        global_step = 0
        
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                if global_step >= max_steps:
                    break
                
                loss = self.training_step(batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1
        
        progress_bar.close()
        print("Training completed!")
    
    def training_step(self, batch):
        """Single training step"""
        
        # Get batch data
        pixel_values = batch['pixel_values'].to(self.device)
        masked_images = batch['masked_image'].to(self.device)
        masks = batch['mask'].to(self.device)
        prompts = batch['prompt']
        
        batch_size = pixel_values.shape[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            masked_latents = self.vae.encode(masked_images).latent_dist.sample()
            masked_latents = masked_latents * self.vae.config.scaling_factor
        
        # Resize masks to latent space (64x64 for 512x512 images)
        masks_resized = F.interpolate(masks, size=(latents.shape[-2], latents.shape[-1]))
        
        # Get text embeddings
        with torch.no_grad():
            text_input_ids = simple_tokenize(self.tokenizer, prompts[0])  # Use first prompt
            text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
            # Expand to match batch size
            text_embeddings = text_embeddings.expand(batch_size, -1, -1)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=self.device).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # For inpainting model: concatenate noisy_latents, mask, and masked_latents
        # Expected input shape: [batch_size, 9, height, width]
        # 4 channels (noisy_latents) + 1 channel (mask) + 4 channels (masked_latents)
        noisy_latents_input = torch.cat([noisy_latents, masks_resized, masked_latents], dim=1)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents_input, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def save_lora_weights(self, output_dir):
        """Save LoRA weights"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get LoRA state dict
        lora_state_dict = get_peft_model_state_dict(self.unet)
        
        # Save weights
        torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.pth"))
        print(f"LoRA weights saved to {output_dir}")

def main():
    # Initialize trainer
    trainer = SimpleDreamBoothTrainer(
        model_name="stabilityai/stable-diffusion-2-inpainting", 
        rank=8,
        learning_rate=1e-4
    )
    
    # Train
    trainer.train(
        data_dir="./data", 
        prompt="a photo of an ambulance vehicle",
        num_epochs=1,
        batch_size=1,
        max_steps=500
    )
    
    # Save weights
    trainer.save_lora_weights("./output/simple_lora")

if __name__ == "__main__":
    main()


# replace the diffusion weights with my lora_weights and see how it's inpainting: baseline