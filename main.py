#!/usr/bin/env python3
"""
Inkpunk Diffusion Implementation - Main Entry Point
"""

from ast import Dict
import os
import torch
from PIL import Image
from transformers import CLIPTokenizer

from src.models.model_loader import preload_models_from_standard_weights
from src.pipeline import generate


def get_device():
    """Automatically detect the best available device."""
    ALLOW_CUDA = True
    ALLOW_MPS = True
    
    if torch.cuda.is_available() and ALLOW_CUDA:
        device = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return device


def load_models(device):
    """Load all required models."""
    print("📦 Loading models...")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer(
        "assets/tokenizer/vocab.json", 
        merges_file="assets/tokenizer/merges.txt"
    )
    
    # Load model weights
    model_file = "assets/Inkpunk-Diffusion-v2.ckpt"
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("Please run 'python download_weights.py' first to download the model weights.")
        return None, None
    
    models = preload_models_from_standard_weights(model_file, device)
    
    print("✅ Models loaded successfully!")
    return models, tokenizer


def text_to_image():
    """Text-to-image generation example."""
    print("\n🎨 Text-to-Image Generation")
    
    # Get device
    device = get_device()
    
    # Load models
    models, tokenizer = load_models(device)
    if not models:
        return
    

    PROMPTS = {
        "inkpunk_13": {
            "pos": "nvinkpunk skull face surfing, sunset sky, rainbow graffiti background, psychedelic spray paint colors, highly detailed",
            "neg": "deformed hands, picture on wall, borders, powerlines, blurry, deformed legs, extra fingers, low-res"
        },
        "inkpunk_14": {
            "pos": "nvinkpunk cyberpunk samurai with neon mask, glowing swords, graffiti wall background, rainbow smoke, vibrant spray paint textures, ultra detailed",
            "neg": "blurry, low resolution, disfigured face, cropped body, extra fingers, dull colors"
        },
        "inkpunk_15": {
            "pos": "nvinkpunk neon biker, chrome skull helmet, rain-soaked alley, rainbow graffiti splashes, cinematic backlight, ultra detailed, vibrant spray paint textures",
            "neg": "blurry, low-res, extra fingers, deformed hands, cropped limbs, dull colors, watermark, text artifacts"
        },
        "inkpunk_16": {
            "pos": "nvinkpunk cyber raven, glowing circuitry feathers, night city skyline, rainbow smoke, graffiti tags, high contrast, ultra detailed",
            "neg": "blurry eyes, double pupils, disfigured beak, jpeg artifacts, posterization, banding, low contrast"
        },
        "inkpunk_17": {
            "pos": "nvinkpunk street samurai, neon oni mask, dual katana, holographic stickers, rainbow paint drips, dynamic pose, ultra detailed",
            "neg": "extra arms, broken anatomy, messy background borders, motion blur, low-res"
        },
        "inkpunk_18": {
            "pos": "nvinkpunk chrome dragon coiled around antenna tower, lightning, rainbow haze, aerosol paint, sharp highlights, ultra detailed",
            "neg": "swirled patterns, muddy colors, deformed claws, text overlay, frame borders"
        },
        "inkpunk_19": {
            "pos": "nvinkpunk synthwave skateboarder, glowing wheels, tunnel of graffiti light, rainbow flare, crisp silhouette, ultra detailed",
            "neg": "cropped feet, extra board, duplicated limbs, compression artifacts, out of frame"
        },
        "inkpunk_20": {
            "pos": "nvinkpunk mecha kitsune, neon tail trails, urban rooftop, rainbow spray, volumetric fog, ultra detailed",
            "neg": "extra tails, deformed muzzle, noisy edges, low detail fur"
        },
        "inkpunk_21": {
            "pos": "nvinkpunk cyber koi swirling in midair, holographic water, rainbow reflections, ink splatter, ultra detailed",
            "neg": "flat shading, sticker-like outline, deformed fins, bland palette"
        },
        "inkpunk_22": {
            "pos": "nvinkpunk glitch monk, floating beads, neon halo, graffiti wall scripture, rainbow smoke, dramatic rim light, ultra detailed",
            "neg": "face distortion, asymmetrical eyes, double nose, low-res"
        },
        "inkpunk_23": {
            "pos": "nvinkpunk neon taxi drifting in rain, chrome reflections, rainbow streaks, street tags, cinematic, ultra detailed",
            "neg": "warped wheels, melted chrome, motion smear, muddy puddles"
        },
        "inkpunk_24": {
            "pos": "nvinkpunk cyber tiger roaring, paint splash mane, rainbow sparks, dark alley shrine, ultra detailed",
            "neg": "extra teeth, distorted jaw, cross-eye, washed out colors"
        },
        "inkpunk_25": {
            "pos": "nvinkpunk hacker shrine, floating keyboards, rainbow cables, CRT glow, graffiti calligraphy, ultra detailed",
            "neg": "generic UI overlays, unreadable text blocks, banding, chromatic noise"
        },
        "inkpunk_26": {
            "pos": "nvinkpunk neon jellyfish city, bioluminescent tendrils, rainbow mist, aerosol dots, ultra detailed",
            "neg": "duplicated tendrils, watery blur, plastic look, low contrast"
        },
        "inkpunk_27": {
            "pos": "nvinkpunk chrome raven skull totem, rainbow flares, shattered glass, paint drips, high microcontrast, ultra detailed",
            "neg": "overexposed highlights, blown whites, thick borders, text"
        },
        "inkpunk_28": {
            "pos": "nvinkpunk desert racer hoverbike, sand neon trail, rainbow heat haze, graffiti decals, cinematic, ultra detailed",
            "neg": "soft focus, mushy edges, duplicated handlebars, warped geometry"
        },
        "inkpunk_29": {
            "pos": "nvinkpunk cyber ballerina spin, hologram tutu, rainbow confetti spray, stage graffiti, crisp silhouette, ultra detailed",
            "neg": "broken ankles, extra fingers, tangled limbs, motion ghosting"
        },
        "inkpunk_30": {
            "pos": "nvinkpunk shrine fox mask vendor, lantern neon, rainbow smoke, spray paint textures, rain sparkle, ultra detailed",
            "neg": "flat lighting, muddy blacks, cropped face, extra hands, watermark"
        }
    }
    
    # Convert to list for easier indexing
    prompt_keys = list(PROMPTS.keys())
    
    print("\n📝 Available Inkpunk Style Prompts:")
    for i, key in enumerate(prompt_keys):
        prompt_preview = PROMPTS[key]['pos'][:80] + "..." if len(PROMPTS[key]['pos']) > 80 else PROMPTS[key]['pos']
        print(f"  {i+1:2d}. {prompt_preview}")
    
    # Let user choose or enter custom prompt
    choice = input(f"\nChoose a prompt (1-{len(PROMPTS)}) or enter 'c' for custom: ").strip()
    
    selected_prompt_key = None
    if choice.lower() == 'c':
        print("\n📝 Creating custom Inkpunk prompt...")
        prompt = input("Enter your positive prompt (tip: start with 'nvinkpunk' for best Inkpunk style): ").strip()
        # Add nvinkpunk prefix if not present
        if not prompt.lower().startswith('nvinkpunk'):
            prompt = f"nvinkpunk {prompt}"
        
        # Get negative prompt from user
        uncond_prompt = input("Enter your negative prompt (what to avoid): ").strip()
        if not uncond_prompt:
            # Use default negative prompt if user enters nothing
            uncond_prompt = "blurry, low-res, deformed, extra fingers, dull colors, watermark, text, bad anatomy"
    else:
        try:
            prompt_idx = int(choice) - 1
            if 0 <= prompt_idx < len(prompt_keys):
                selected_prompt_key = prompt_keys[prompt_idx]
                prompt = PROMPTS[selected_prompt_key]['pos']
                uncond_prompt = PROMPTS[selected_prompt_key]['neg']
            else:
                print("❌ Invalid choice, using default prompt.")
                selected_prompt_key = prompt_keys[0]
                prompt = PROMPTS[selected_prompt_key]['pos']
                uncond_prompt = PROMPTS[selected_prompt_key]['neg']
        except ValueError:
            print("❌ Invalid input, using default prompt.")
            selected_prompt_key = prompt_keys[0]
            prompt = PROMPTS[selected_prompt_key]['pos']
            uncond_prompt = PROMPTS[selected_prompt_key]['neg']
    
    print(f"\n🎯 Generating Inkpunk style image...")
    print(f"📋 Positive prompt: {prompt}")
    print(f"🚫 Negative prompt: {uncond_prompt}")
    
    # Generation parameters optimized for Inkpunk style
    do_cfg = True
    cfg_scale = 9.0  # Slightly higher for more prompt adherence
    sampler = "ddpm"
    num_inference_steps = 80  # Good balance between quality and speed
    seed = 42
    
    # Allow user to modify seed
    seed_input = input(f"\nEnter seed (current: {seed}, press Enter to keep): ").strip()
    if seed_input.isdigit():
        seed = int(seed_input)
    
    # Generate image
    print("🚀 Starting generation...")
    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=None,
        strength=0.8,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    
    # Save generated image with descriptive filename
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with prompt info
    if selected_prompt_key:
        filename = f"inkpunk_{selected_prompt_key}_seed_{seed}.png"
    else:
        filename = f"inkpunk_custom_seed_{seed}.png"
    
    output_path = os.path.join(output_dir, filename)
    Image.fromarray(output_image).save(output_path)
    
    print(f"✅ Inkpunk style image generated and saved to: {output_path}")
    print(f"🎨 Used prompt: {prompt[:100]}...")


def image_to_image():
    """Image-to-image generation example."""
    print("\n🖼️ Image-to-Image Generation")
    
    # Get input image path
    image_path = input("Enter path to input image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Get device
    device = get_device()
    
    # Load models
    models, tokenizer = load_models(device)
    if not models:
        return
    
    # Load input image
    try:
        input_image = Image.open(image_path).convert('RGB')
        print(f"📸 Loaded image: {image_path}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Get prompt
    prompt = input("Enter transformation prompt: ").strip()
    
    print(f"\n🎯 Transforming image with: {prompt}")
    
    # Generation parameters
    uncond_prompt = ""
    do_cfg = True
    cfg_scale = 7.5
    strength = 0.8  # How much to change the input image
    sampler = "ddpm"
    num_inference_steps = 80
    seed = 42
    
    # Generate image
    print("🚀 Starting transformation...")
    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    
    # Save generated image
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"transformed_image_seed_{seed}.png")
    Image.fromarray(output_image).save(output_path)
    
    print(f"✅ Transformed image saved to: {output_path}")


def main():
    """Main application entry point."""
    print("🎨 Stable Diffusion v1.5 - PyTorch Implementation")
    print("=" * 50)
    
    while True:
        print("\n📋 Available options:")
        print("  1. Text-to-Image Generation")
        print("  2. Image-to-Image Generation")
        print("  3. Download Model Weights")
        print("  4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            text_to_image()
        elif choice == "2":
            image_to_image()
        elif choice == "3":
            print("\n📥 To download model weights, run:")
            print("python download_weights.py")
            print("\nMake sure you have a .env file with your HUGGINGFACE_TOKEN")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
