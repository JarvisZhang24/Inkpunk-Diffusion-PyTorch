# Inkpunk-Diffusion-PyTorch

A clean, modular PyTorch implementation of Inkpunk Diffusion, based on the Stable Diffusion v1.5 architecture. This project provides a clear and understandable codebase for generating images in the distinctive Inkpunk artistic style.

## Features

- **Inkpunk Style Generation**: Create unique images with the iconic Inkpunk aesthetic.
- **Text-to-Image Generation**: Generate images from text prompts.
- **Image-to-Image Generation**: Transform existing images using text prompts.
- **Modular Architecture**: Clean separation of components (VAE, UNET, CLIP, Diffusion).
- **Automatic Device Detection**: Supports CUDA, MPS (Apple Silicon), and CPU.
- **Weight Management**: Includes scripts for easy weight management.

## Project Structure

```
Stable_diffusion/
├── src/
│   ├── models/
│   │   ├── attention.py      # Self & Cross attention mechanisms
│   │   ├── vae.py           # VAE Encoder & Decoder
│   │   ├── clip.py          # CLIP text encoder
│   │   ├── unet.py          # UNET denoising network
│   │   ├── diffusion.py     # Diffusion model & DDPM sampler
│   │   └── model_loader.py  # Weight loading & conversion
│   └── pipeline.py          # Main generation pipeline
├── assets/
│   └── Inkpunk-Diffusion-v2.ckpt # Model checkpoint
├── main.py                  # Interactive main application
├── download_weights.py      # Base weight download script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

1.  **Clone and navigate to the project:**
    ```bash
    cd Stable_diffusion
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download model weights:**
    This project is configured to use `Inkpunk-Diffusion-v2.ckpt`. Make sure you have this file in the `assets/` directory. If you need the base Stable Diffusion v1.5 weights for other purposes, you can use the provided script after setting up your Hugging Face token in a `.env` file.
    ```bash
    # (Optional) For base SD 1.5 weights
    # cp .env.example .env 
    # python download_weights.py
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

## Usage

### Interactive Mode
Run `python main.py` for an interactive interface. The script is pre-configured to load the Inkpunk Diffusion model.

### Programmatic Usage
```python
from transformers import CLIPTokenizer
from src.models.model_loader import preload_models_from_standard_weights
from src.pipeline import generate

# Load models
device = "cuda"  # or "mps", "cpu"
tokenizer = CLIPTokenizer("assets/tokenizer/vocab.json", 
                         merges_file="assets/tokenizer/merges.txt")
# Ensure you are loading the correct Inkpunk checkpoint
models = preload_models_from_standard_weights("assets/Inkpunk-Diffusion-v2.ckpt", device)

# Generate image
output_image = generate(
    prompt="A cyberpunk rogue in a futuristic city, inkpunk style, intricate details, manga aesthetic",
    models=models,
    device=device,
    tokenizer=tokenizer,
    seed=42
)
# The output image will be saved in the outputs/ directory
```

## Configuration

### Generation Parameters
- `prompt`: Text description of desired image
- `uncond_prompt`: Negative prompt (what to avoid)
- `cfg_scale`: Classifier-free guidance scale (1-20, default: 7.5)
- `n_inference_steps`: Number of denoising steps (20-200, default: 50)
- `seed`: Random seed for reproducible results
- `strength`: For img2img, how much to change input (0-1, default: 0.8)

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU (very slow)
- **Recommended**: 8GB+ VRAM CUDA GPU or Apple Silicon Mac
- **Storage**: ~4GB for model weights

## Model Architecture

This implementation includes:

- **VAE**: Variational Autoencoder for image encoding/decoding
- **CLIP**: Text encoder for processing prompts
- **UNET**: Denoising network with attention mechanisms
- **DDPM Sampler**: Denoising Diffusion Probabilistic Model sampler

All components are implemented from scratch in PyTorch with detailed comments.

## Weight Conversion

The `load_from_standard_weights` function converts original Stable Diffusion compatible checkpoint weights (like Inkpunk Diffusion) to our custom model format. This ensures compatibility while maintaining the exact same model behavior as the original implementation.

## License

This project is for educational purposes. The Inkpunk Diffusion model weights are subject to their original license. Base Stable Diffusion model weights are subject to the license from Stability AI.

## Troubleshooting

**Memory Issues**: Reduce `n_inference_steps` or image dimensions.

**Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
