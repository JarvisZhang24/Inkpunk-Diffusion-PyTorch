# Stable Diffusion v1.5 - PyTorch Implementation

A clean, modular PyTorch implementation of Stable Diffusion v1.5 converted from Jupyter notebook format.

## Features

- **Text-to-Image Generation**: Generate images from text prompts
- **Image-to-Image Generation**: Transform existing images using text prompts  
- **Modular Architecture**: Clean separation of components (VAE, UNET, CLIP, Diffusion)
- **Automatic Device Detection**: Supports CUDA, MPS (Apple Silicon), and CPU
- **Weight Management**: Automatic download and conversion of Hugging Face weights

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
├── main.py                  # Interactive main application
├── download_weights.py      # Weight download script
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd Stable_diffusion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

4. **Download model weights:**
   ```bash
   python download_weights.py
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Interactive Mode
Run `python main.py` for an interactive interface with options for:
- Text-to-image generation with sample prompts
- Image-to-image transformation
- Weight download management

### Programmatic Usage
```python
from transformers import CLIPTokenizer
from src.models.model_loader import preload_models_from_standard_weights
from src.pipeline import generate

# Load models
device = "cuda"  # or "mps", "cpu"
tokenizer = CLIPTokenizer("assets/tokenizer/vocab.json", 
                         merges_file="assets/tokenizer/merges.txt")
models = preload_models_from_standard_weights("assets/v1-5-pruned-emaonly.ckpt", device)

# Generate image
output_image = generate(
    prompt="A beautiful sunset over mountains",
    models=models,
    device=device,
    tokenizer=tokenizer,
    seed=42
)
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

The `load_from_standard_weights` function converts original Stable Diffusion checkpoint weights to our custom format. This ensures compatibility while maintaining the exact same model behavior as the original implementation.

## License

This project is for educational purposes. Model weights are subject to the original Stable Diffusion license from Stability AI.

## Troubleshooting

**Memory Issues**: Reduce `n_inference_steps` or use CPU with smaller batch sizes

**Download Fails**: Check your Hugging Face token and internet connection

**Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
