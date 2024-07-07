from diffusers import DiffusionPipeline
import torch

# Modeli indirin
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",safety_checker=None,)

# Modeli lokal bir dizine kaydedin
model_path = "./stabilityai/stable-diffusion-2-1-base"
pipeline.save_pretrained(model_path)
