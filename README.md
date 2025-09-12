# Stable-Diffusion-CPU-Ultimate-Optimized-txt2img-img2img

# CPU-Optimized Stable Diffusion with Multiple LoRA Support and Image-to-Image

This project provides a highly optimized implementation of Stable Diffusion designed specifically for CPU execution with advanced features. It enables users to generate high-quality images using limited hardware resources (without requiring a GPU), making AI image generation accessible on virtually any computer.

## Key Features

✅ **CPU Memory Optimization**  
- Reduces memory consumption to 1.0-1.5 GB using 8-bit quantization techniques
- Works on systems with limited resources (as low as 2 GB RAM)
- Implements attention slicing and VAE slicing for additional memory savings

✅ **Multiple LoRA Support**  
- Load and apply multiple LoRA weights simultaneously (1-5 LoRAs)
- Individual strength control for each LoRA (0.1-1.5)
- Automatic LoRA application without requiring special tags in prompts (`<lora:lora_name>`)

✅ **Image-to-Image Conversion**  
- Precise implementation with exact latent size handling
- VAE-only quantization to preserve image quality
- Adjustable transformation strength (0.1-1.0)

✅ **Advanced Configuration Options**  
- Multiple scheduler options: DDIM, PNDMScheduler, LMS, Euler, Euler Ancestral, DPM++, LCM, UniPC
- Negative prompt support
- Dynamic CFG scale adjustment (1.0-20.0)
- Multiple preset configurations for common styles

✅ **Style Presets**  
- Photorealistic (high-quality photos)
- Anime/Cartoon style
- Digital Art (concept art, illustrations)
- 3D Render style

✅ **Additional Features**  
- Save and load configuration profiles for reuse
- Post-processing enhancements (contrast, sharpness, color)
- Detailed generation reports
- Prompt history tracking
- Memory usage monitoring

## System Requirements

- **Minimum RAM:** 1.5 GB (for base models)
- **Processor:** Minimum 2 cores (generation speed depends on core count)
- **Disk Space:** Approximately 2 GB for models and dependencies
- **Operating System:** Windows, macOS, or Linux

## Installation and Setup

1. Install required dependencies:
```bash
pip install torch diffusers transformers accelerate safetensors psutil tqdm pillow
```

2. Download required model files:
   - Default models: `ds_lcm.safetensors` or `wm_lcm.safetensors`
   - Place models in the same directory as the script

3. Run the application:
```bash
python stable_diffusion_cpu.py
```

## Usage Guide

1. **Select Generation Mode:**
   - **Text-to-Image:** Generate images from text prompts
   - **Image-to-Image:** Modify existing images

2. **Configure LoRA (Optional):**
   - Choose whether to use LoRA for style enhancement
   - Specify 1-5 LoRA files with individual strength settings (0.1-1.5)

3. **Select Configuration:**
   - **Quick Presets:** Choose from pre-configured style templates
   - **Custom Configuration:** Manually set all parameters

4. **Scheduler Selection:**
   - Choose from 8 different scheduler options
   - Configure CFG scale (guidance strength)
   - Note: LCM scheduler works best with low CFG scale (1.0-2.0)

5. **Enter Prompts:**
   - Main prompt describing the desired image
   - Optional negative prompt to exclude unwanted elements
   - Support for prompt history (last 50 prompts)

6. **Image Dimensions (Text-to-Image):**
   - Select from common aspect ratios or enter custom dimensions
   - Dimensions must be multiples of 8 (automatically adjusted if needed)

7. **Image-to-Image Specifics:**
   - Provide path to input image
   - Set transformation strength (0.1-1.0, default 0.75)

8. **Post-Generation Actions:**
   - View generated image
   - Apply post-processing enhancements
   - Save as high-quality JPEG
   - Generate detailed report
   - Save current configuration for future use

## Technical Notes

- This implementation is specifically optimized for CPU execution
- LoRAs are applied programmatically without requiring special tags in prompts
- Generation time depends on CPU core count (approximately 5-10 minutes for 512x512 on a 4-core CPU)
- For low-memory systems, reduce image size to 384x384 or 256x256
- Image-to-Image mode disables VAE quantization to preserve quality
- LCM scheduler requires low CFG scale (1.0-2.0) for best results
- All dimensions are automatically adjusted to multiples of 8 for compatibility

## License

This project is released under the **YSNRFD LICENSE**.

## Developer Contact

- Telegram: [@ysnrfd](https://t.me/ysnrfd)
- GitHub: [ysnrfd](https://github.com/ysnrfd)
- Hugging Face: [ysnrfd](https://huggingface.co/ysnrfd)


**Note:** This implementation is designed for systems without GPU access, allowing quality image generation using only CPU resources. It offers better performance compared to standard implementations without memory optimizations.
```
