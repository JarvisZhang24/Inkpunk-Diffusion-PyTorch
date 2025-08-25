## Inkpunk Diffusion (PyTorch)

A from-scratch, modular PyTorch re-implementation of an Inkpunk-style Stable Diffusion v1.5 pipeline. I built this to demonstrate hands-on expertise with diffusion models, attention-based UNet **architectures**, CLIP text encoding, and VAE image compression — with clean, readable code and reproducible results across CUDA, Apple Silicon (MPS), and CPU.

### What this project demonstrates

- End-to-end diffusion pipeline: CLIP tokenizer/encoder → UNet denoiser → VAE decode
- Weight compatibility: load SD-compatible checkpoints (e.g., Inkpunk) into custom modules
- Deterministic, reproducible generation via seed control and fixed samplers
- Practical engineering: device auto-detection, modular design, and simple CLI

### Tech stack

- Python, PyTorch, PIL
- Hugging Face `transformers` (CLIPTokenizer)
- CUDA / Apple Metal (MPS) / CPU

---

## Results Gallery (Inkpunk style)

Below are sample generations produced by the codebase using curated Inkpunk prompts. Each image is reproducible with the listed prompt and seed.


![Inkpunk 2 — seed 42](outputs/inkpunk_2_seed_42.png)

Prompt (positive):
> nvinkpunk cyberpunk samurai with neon mask, glowing swords, graffiti wall background, rainbow smoke, vibrant spray paint textures, ultra detailed

Negative prompt:
> flat shading, sticker-like outline, deformed fins, bland palette

![Inkpunk 13 — seed 42](outputs/inkpunk_13_seed_42.png)

Prompt (positive):
> nvinkpunk hacker shrine, floating keyboards, rainbow cables, CRT glow, graffiti calligraphy, ultra detailed

Negative prompt:
> generic UI overlays, unreadable text blocks, banding, chromatic noise

![Inkpunk 14 — seed 42](outputs/inkpunk_14_seed_42.png)

Prompt (positive):
> nvinkpunk neon jellyfish city, bioluminescent tendrils, rainbow mist, aerosol dots, ultra detailed

Negative prompt:
> duplicated tendrils, watery blur, plastic look, low contrast

![Inkpunk 16 — seed 42](outputs/inkpunk_16_seed_42.png)

Prompt (positive):
> nvinkpunk desert racer hoverbike, sand neon trail, rainbow heat haze, graffiti decals, cinematic, ultra detailed

Negative prompt:
> soft focus, mushy edges, duplicated handlebars, warped geometry

To add more results, generate with your preferred prompts (see Reproduce My Results) and place the images under `outputs/`. Then embed them here in the same way.

---

## Quickstart

1. Navigate to the project root:
    ```bash
    cd Stable_diffusion
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Weights
    - This project expects `assets/Inkpunk-Diffusion-v2.ckpt` to be present (already included in my setup).
    - If you need to fetch base SD 1.5 weights for experimentation, configure your Hugging Face token and use the helper script:
      ```bash
      # optional
      # cp .env 
      # python download_weights.py
      ```

4. Run the interactive app:
    ```bash
    python main.py
    ```

---

## Reproduce My Results

### Interactive (recommended)
- Run `python main.py` and select “Text-to-Image Generation”.
- Pick a built-in Inkpunk prompt or enter your own. Use seed `42` to match my gallery.

### Programmatic usage
```python
from transformers import CLIPTokenizer
from src.models.model_loader import preload_models_from_standard_weights
from src.pipeline import generate

device = "cuda"  # or "mps", "cpu"
tokenizer = CLIPTokenizer(
    "assets/tokenizer/vocab.json",
    merges_file="assets/tokenizer/merges.txt"
)

models = preload_models_from_standard_weights(
    "assets/Inkpunk-Diffusion-v2.ckpt", device
)

image = generate(
    prompt=(
        "nvinkpunk neon taxi drifting in rain, chrome reflections, "
        "rainbow streaks, street tags, cinematic, ultra detailed"
    ),
    uncond_prompt=(
        "warped wheels, melted chrome, motion smear, muddy puddles"
    ),
    models=models,
    device=device,
    tokenizer=tokenizer,
    seed=42,
    do_cfg=True,
    cfg_scale=9.0,
    sampler_name="ddpm",
    n_inference_steps=80,
)
# Save to outputs/ using your preferred filename
```

Example additional curated prompts (InkPunk style):
- `nvinkpunk cyber koi swirling in midair, holographic water, rainbow reflections, ink splatter, ultra detailed`  
  Negative: `flat shading, sticker-like outline, deformed fins, bland palette`
- `nvinkpunk hacker shrine, floating keyboards, rainbow cables, CRT glow, graffiti calligraphy, ultra detailed`  
  Negative: `generic UI overlays, unreadable text blocks, banding, chromatic noise`

---

## Architecture Overview

- VAE: Variational Autoencoder for latent-space image compression and decoding
- CLIP: Text encoder (tokenization + embeddings) for prompt conditioning
- UNet: Denoising network with self- and cross-attention blocks
- DDPM sampler: Iterative denoising loop (configurable steps and guidance scale)

All components are implemented in PyTorch with a clean, modular design for readability and extension.

## Technical Pipeline

The following diagram illustrates the complete text-to-image generation pipeline, showing how each component transforms data through the system:

```mermaid
graph TD
    A["Text Input<br/>nvinkpunk cyberpunk samurai"] --> B["CLIP Tokenizer<br/>Text→Token IDs<br/>(1,77)"]
    
    B --> C["CLIP Embedding<br/>Token+Position Embeddings<br/>(1,77,768)"]
    
    C --> D["12-Layer CLIP Processing<br/>Self-Attention+Feed-Forward<br/>(1,77,768)"]
    
    D --> E["Text Context Vector<br/>Positive+Negative Prompts<br/>(2,77,768)"]
    
    F["Random Noise Generation<br/>Standard Normal Distribution<br/>(1,4,64,64)"] --> G["DDPM Sampler Init<br/>80 Timesteps<br/>999→0"]
    
    G --> H["Iterative Denoising Loop<br/>80 Iterations"]
    
    H --> I["Time Embedding<br/>Sinusoidal Encoding<br/>(1,320)"]
    
    I --> J["Time Information Expansion<br/>(1,320)→(1,1280)"]
    
    E --> K["UNet Encoder Path<br/>Downsampling+Attention<br/>64×64→8×8"]
    
    J --> K
    F --> K
    
    K --> L["UNet Bottleneck<br/>Deep Feature Extraction<br/>(1,1280,8,8)"]
    
    L --> M["UNet Decoder Path<br/>Upsampling+Skip Connections<br/>8×8→64×64"]
    
    M --> N["Noise Prediction<br/>UNet Output Layer<br/>(1,4,64,64)"]
    
    N --> O["Classifier-Free Guidance<br/>Conditional vs Unconditional<br/>Noise Mixing"]
    
    O --> P["DDPM Sampling Step<br/>Remove Predicted Noise<br/>Update Latent Vector"]
    
    P --> Q{"Completed<br/>80 Steps?"}
    
    Q -->|No| H
    Q -->|Yes| R["Final Latent Representation<br/>Fully Denoised<br/>(1,4,64,64)"]
    
    R --> S["VAE Decoder<br/>Latent Space→Image Space<br/>(1,4,64,64)→(1,3,512,512)"]
    
    S --> T["Post-processing<br/>Normalization+Format Conversion<br/>(-1,1)→(0,255)"]
    
    T --> U["Final Output Image<br/>512×512 RGB<br/>Inkpunk Samurai Artwork"]

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#fff3e0
    style R fill:#e8f5e8
    style U fill:#ffebee
```

### Key Pipeline Stages:

1. **Text Processing**: CLIP transforms natural language into semantic embeddings
2. **Noise Initialization**: Pure Gaussian noise in compressed latent space (8× smaller than image space)
3. **Iterative Denoising**: UNet progressively removes noise guided by text context over 80 steps
4. **Image Reconstruction**: VAE decoder converts latent representation back to RGB image

For a detailed technical walkthrough, see [Inkpunk_Diffusion_Technical_Overview.md](Inkpunk_Diffusion_Technical_Overview.md).

---

## Project **Structure**

```
Inkpunk_Diffusion/
├── src/
│   ├── models/
│   │   ├── attention.py      # Self & cross-attention
│   │   ├── vae.py            # VAE encoder & decoder
│   │   ├── clip.py           # CLIP text encoder
│   │   ├── unet.py           # UNet denoiser
│   │   ├── diffusion.py      # Diffusion loop & DDPM sampler
│   │   └── model_loader.py   # Load/convert SD-compatible weights
│   └── pipeline.py           # Orchestration for generation
├── assets/
│   └── Inkpunk-Diffusion-v2.ckpt # Inkpunk weights
│   └── tokenizer/               # CLIP tokenizer 
│       └── merges.txt           # Merges file
│       └── vocab.json           # Vocab file
├── .env                         # Environment variables(Hugging Face token)
├── outputs/                     # Generated images
├── main.py                   # Interactive CLI
├── download_weights.py       # Optional weights helper
├── requirements.txt          # Dependencies
├── Inkpunk_Diffusion_Technical_Overview.md # Technical Overview
└── README.md
```

---

## License

This repository is for educational and portfolio purposes. The Inkpunk Diffusion model weights are governed by their original license. Base Stable Diffusion weights are governed by Stability AI’s license.

---

## Troubleshooting

- Memory constraints: lower `n_inference_steps` or image size; prefer GPU/MPS when available
- Import/runtime errors: ensure dependencies are installed via `pip install -r requirements.txt`
