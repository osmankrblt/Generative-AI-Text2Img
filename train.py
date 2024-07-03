#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module



ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-08
ADAM_WEIGHT_DECAY = 1e-2
ACCELERATOR_PROJECT_CONFIG = ""
ALLOW_TF32 = True
BITSANDBYTES = ""
CACHE_DIR = None
CAPTION_COLUMN="gpt_description"
CENTER_CROP = True
CHECKPOINTING_STEPS = 500
CHECKPOINTS_TOTAL_LIMIT = None
DATA_FILES = ""
DATASET_CONFIG_NAME = None
DATASET_NAME = None
DATALOADER_NUM_WORKERS = 0
DREAM_DETAIL_PRESERVATION = 1.0
DREAM_TRAINING = True
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = True
FOREACH_EMA = True
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
INPUT_PERTURBATION=""
IMAGE_COLUMN="filename"
LEARNING_RATE = 1e-4
LOGGING_DIR = "LOGGING"
LR_WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
MAX_TRAIN_SAMPLES = None
MIXED_PRECISION = None
NON_EMA_REVISION = None
NOISE_OFFSET = 0
NUM_TRAIN_EPOCHS = 100
OFFLOAD_EMA = True
OUTPUT_DIR = "sd-model-finetuned"
PREDICTION_TYPE = None
PRETRAINED_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
RANDOM_FLIP = True
REPORT_TO = "wandb"
RESOLUTION = 1024
REVISION = None
SCALE_LR = False
SEED = None
SNR_GAMMA = None
TRAIN_BATCH_SIZE = 16
TRAIN_DATA_DIR = "./Datasets/Russian Paintings/data/"
TRACKER_PROJECT_NAME = "text2image-fine-tune"
USE_8BIT_ADAM = True
USE_EMA = True
VALIDATION_EPOCHS = 5
VALIDATION_PROMPTS = "Isaac Ilyich Levittan woman playing snowman"
VALIDATION_PROMPTS = None
VARIANT = None

TRACKER_CONFIG = {
    "ADAM_BETA1": ADAM_BETA1,
    "ADAM_BETA2": ADAM_BETA2,
    "ADAM_EPSILON": ADAM_EPSILON,
    "ADAM_WEIGHT_DECAY": ADAM_WEIGHT_DECAY,
    "ACCELERATOR_PROJECT_CONFIG": ACCELERATOR_PROJECT_CONFIG,
    "ALLOW_TF32": ALLOW_TF32,
    "BITSANDBYTES": BITSANDBYTES,
    "CACHE_DIR": CACHE_DIR,
    "CENTER_CROP": CENTER_CROP,
    "CHECKPOINTING_STEPS": CHECKPOINTING_STEPS,
    "CHECKPOINTS_TOTAL_LIMIT": CHECKPOINTS_TOTAL_LIMIT,
    "DATA_FILES": DATA_FILES,
    "DATASET_CONFIG_NAME": DATASET_CONFIG_NAME,
    "DATASET_NAME": DATASET_NAME,
    "DATALOADER_NUM_WORKERS": DATALOADER_NUM_WORKERS,
    "DREAM_DETAIL_PRESERVATION": DREAM_DETAIL_PRESERVATION,
    "DREAM_TRAINING": DREAM_TRAINING,
    "ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION": ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION,
    "FOREACH_EMA": FOREACH_EMA,
    "GRADIENT_ACCUMULATION_STEPS": GRADIENT_ACCUMULATION_STEPS,
    "GRADIENT_CHECKPOINTING": GRADIENT_CHECKPOINTING,
    "INPUT_PERTURBATION":INPUT_PERTURBATION,
    "IMAGE_COLUMN":IMAGE_COLUMN,
    "LEARNING_RATE": LEARNING_RATE,
    "LOGGING_DIR": LOGGING_DIR,
    "LR_WARMUP_STEPS": LR_WARMUP_STEPS,
    "MAX_GRAD_NORM": MAX_GRAD_NORM,
    "MAX_TRAIN_SAMPLES": MAX_TRAIN_SAMPLES,
    "MIXED_PRECISION": MIXED_PRECISION,
    "NON_EMA_REVISION": NON_EMA_REVISION,
    "NOISE_OFFSET": NOISE_OFFSET,
    "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,
    "OFFLOAD_EMA": OFFLOAD_EMA,
    "OUTPUT_DIR": OUTPUT_DIR,
    "PREDICTION_TYPE": PREDICTION_TYPE,
    "PRETRAINED_MODEL_NAME_OR_PATH": PRETRAINED_MODEL_NAME_OR_PATH,
    "RANDOM_FLIP": RANDOM_FLIP,
    "REPORT_TO": REPORT_TO,
    "RESOLUTION": RESOLUTION,
    "REVISION": REVISION,
    "SCALE_LR": SCALE_LR,
    "SEED": SEED,
    "SNR_GAMMA": SNR_GAMMA,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TRACKER_PROJECT_NAME": TRACKER_PROJECT_NAME,
    "USE_8BIT_ADAM": USE_8BIT_ADAM,
    "USE_EMA": USE_EMA,
    "VALIDATION_EPOCHS": VALIDATION_EPOCHS,
    "VALIDATION_PROMPTS": VALIDATION_PROMPTS,
    "VARIANT": VARIANT
}



if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(VALIDATION_PROMPTS))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{PRETRAINED_MODEL_NAME_OR_PATH}** on the **{DATASET_NAME}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {VALIDATION_PROMPTS}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{VALIDATION_PROMPTS[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {NUM_TRAIN_EPOCHS}
* Learning rate: {LEARNING_RATE}
* Batch size: {TRAIN_BATCH_SIZE}
* Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}
* Image RESOLUTION: {RESOLUTION}
* Mixed-precision: {MIXED_PRECISION}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=PRETRAINED_MODEL_NAME_OR_PATH,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        REVISION=REVISION,
        VARIANT=VARIANT,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION:
        pipeline.ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION()

    if SEED is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_SEED(SEED)

    images = []
    for i in range(len(VALIDATION_PROMPTS)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(VALIDATION_PROMPTS[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def main():
    
    global LOGGING_DIR
    global MIXED_PRECISION
    global LEARNING_RATE

    
    LOGGING_DIR = os.path.join(OUTPUT_DIR, LOGGING_DIR)

    ACCELERATOR_PROJECT_CONFIG = ProjectConfiguration(project_dir=OUTPUT_DIR, logging_dir=LOGGING_DIR)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision= MIXED_PRECISION,
        log_with=REPORT_TO,
        project_config=ACCELERATOR_PROJECT_CONFIG,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training SEED now.
    if SEED is not None:
        set_seed(SEED)

    # Handle the repository creation
    if accelerator.is_main_process:
        if OUTPUT_DIR is not None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer", REVISION=REVISION
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder", revision=REVISION, variant=VARIANT
        )
        vae = AutoencoderKL.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae", revision=REVISION, variant=VARIANT
        )

    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", revision=NON_EMA_REVISION
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if USE_EMA:
        ema_unet = UNet2DConditionModel.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", revision=REVISION, variant=VARIANT
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=FOREACH_EMA,
        )

    if ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, OUTPUT_DIR):
            if accelerator.is_main_process:
                if USE_EMA:
                    ema_unet.save_pretrained(os.path.join(OUTPUT_DIR, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(OUTPUT_DIR, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if USE_EMA:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=FOREACH_EMA
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if OFFLOAD_EMA:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    

    if ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if SCALE_LR:
        LEARNING_RATE = (
            LEARNING_RATE * GRADIENT_ACCUMULATION_STEPS * TRAIN_BATCH_SIZE * accelerator.num_processes
        )

    # Initialize the optimizer
    if USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install BITSANDBYTES to use 8-bit Adam. You can do so by running `pip install BITSANDBYTES`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if DATASET_NAME is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG_NAME,
            CACHE_DIR=CACHE_DIR,
            data_dir=TRAIN_DATA_DIR,
        )
    else:
        DATA_FILES = {}
        if TRAIN_DATA_DIR is not None:
            DATA_FILES["train"] = os.path.join(TRAIN_DATA_DIR, "**")
        
        dataset = load_dataset(
            path=TRAIN_DATA_DIR,
            
            DATA_FILES=DATA_FILES,
            CACHE_DIR=CACHE_DIR,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(DATASET_NAME, None)
    if IMAGE_COLUMN is None:
        IMAGE_COLUMN = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        IMAGE_COLUMN = IMAGE_COLUMN
        if IMAGE_COLUMN not in column_names:
            raise ValueError(
                f"--IMAGE_COLUMN' value '{IMAGE_COLUMN}' needs to be one of: {', '.join(column_names)}"
            )
    if CAPTION_COLUMN is None:
        CAPTION_COLUMN = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        CAPTION_COLUMN = CAPTION_COLUMN
        if CAPTION_COLUMN not in column_names:
            raise ValueError(
                f"--CAPTION_COLUMN' value '{CAPTION_COLUMN}' needs to be one of: {', '.join(column_names)}"
            )
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[CAPTION_COLUMN]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{CAPTION_COLUMN}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION) if CENTER_CROP else transforms.RandomCrop(RESOLUTION),
            transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[IMAGE_COLUMN]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if MAX_TRAIN_SAMPLES is not None:
            dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=DATALOADER_NUM_WORKERS,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    if max_train_steps is None:
        max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if USE_EMA:
        if OFFLOAD_EMA:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.MIXED_PRECISION == "fp16":
        weight_dtype = torch.float16
        MIXED_PRECISION = accelerator.MIXED_PRECISION
    elif accelerator.MIXED_PRECISION == "bf16":
        weight_dtype = torch.bfloat16
        MIXED_PRECISION = accelerator.MIXED_PRECISION

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    if overrode_max_train_steps:
        max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    NUM_TRAIN_EPOCHS = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
      
        TRACKER_CONFIG.pop("VALIDATION_PROMPTS")
        accelerator.init_trackers(TRACKER_PROJECT_NAME, TRACKER_CONFIG)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(OUTPUT_DIR)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(OUTPUT_DIR, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, NUM_TRAIN_EPOCHS):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if NOISE_OFFSET:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += NOISE_OFFSET * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if INPUT_PERTURBATION:
                    new_noise = noise + INPUT_PERTURBATION * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if INPUT_PERTURBATION:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if PREDICTION_TYPE is not None:
                    # set PREDICTION_TYPE of scheduler if defined
                    noise_scheduler.register_to_config(PREDICTION_TYPE=PREDICTION_TYPE)

                if noise_scheduler.config.PREDICTION_TYPE == "epsilon":
                    target = noise
                elif noise_scheduler.config.PREDICTION_TYPE == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.PREDICTION_TYPE}")

                if DREAM_TRAINING:
                    noisy_latents, target = compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        encoder_hidden_states,
                        DREAM_DETAIL_PRESERVATION,
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if SNR_GAMMA is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, SNR_GAMMA * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.PREDICTION_TYPE == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.PREDICTION_TYPE == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(TRAIN_BATCH_SIZE)).mean()
                train_loss += avg_loss.item() / GRADIENT_ACCUMULATION_STEPS

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if USE_EMA:
                    if OFFLOAD_EMA:
                        ema_unet.to(device="cuda", non_blocking=True)
                    ema_unet.step(unet.parameters())
                    if OFFLOAD_EMA:
                        ema_unet.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % CHECKPOINTING_STEPS == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `CHECKPOINTS_TOTAL_LIMIT`
                        if CHECKPOINTS_TOTAL_LIMIT is not None:
                            checkpoints = os.listdir(OUTPUT_DIR)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `CHECKPOINTS_TOTAL_LIMIT - 1` checkpoints
                            if len(checkpoints) >= CHECKPOINTS_TOTAL_LIMIT:
                                num_to_remove = len(checkpoints) - CHECKPOINTS_TOTAL_LIMIT + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(OUTPUT_DIR, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            if VALIDATION_PROMPTS is not None and epoch % VALIDATION_EPOCHS == 0:
                if USE_EMA:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    TRACKER_CONFIG,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if USE_EMA:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if USE_EMA:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            REVISION=REVISION,
            VARIANT=VARIANT,
        )
        pipeline.save_pretrained(OUTPUT_DIR)

        # Run a final round of inference.
        images = []
        if VALIDATION_PROMPTS is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION:
                pipeline.ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION()

            if SEED is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_SEED(SEED)

            for i in range(len(VALIDATION_PROMPTS)):
                with torch.autocast("cuda"):
                    image = pipeline(VALIDATION_PROMPTS[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

      

    accelerator.end_training()


if __name__ == "__main__":
    main()