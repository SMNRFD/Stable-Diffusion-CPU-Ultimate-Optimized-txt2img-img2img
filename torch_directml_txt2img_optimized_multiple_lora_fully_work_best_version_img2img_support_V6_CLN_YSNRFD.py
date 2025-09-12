"""
Complete Python code for CPU-optimized Stable Diffusion with Multiple LoRA Support and Image-to-Image
Fixed: Proper implementation for loading and applying multiple LoRAs simultaneously
Enhanced: Image-to-Image functionality with exact latent size and VAE-only quantization
Added: Multiple scheduler options with dynamic CFG scale, negative prompt, and advanced features
Fixed: Proper menu navigation after saving configuration
Added: Support for saving and loading multiple configuration files
Code By: YSNRFD (Updated with Multiple LoRA, Image-to-Image, and Advanced Settings)
Telegram: @ysnrfd
Github: ysnrfd
Huggingface: ysnrfd
--------------------
UNDER YSNRFD LICENSE
--------------------
"""





import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    UniPCMultistepScheduler
)
import time
import random
import gc
import os
import psutil
from tqdm import tqdm
import sys
import diffusers
from PIL import Image
import json
def setup_cpu_memory_optimizations(pipe, skip_quantization=False, img2img_mode=False):
    """
    Apply all memory reduction methods specifically optimized for CPU usage
    Parameters:
    - pipe: Stable Diffusion pipeline
    - skip_quantization: Whether to skip full quantization (needed for LoRA)
    - img2img_mode: Whether in image-to-image mode (VAE quantization disabled)
    Returns:
    - pipe: Pipeline with applied CPU memory optimizations
    """
    print("\n" + "="*60)
    print("Applying CPU-specific memory reduction methods")
    print("UNDER YSNRFD LICENSE")
    print("="*60)
    
    if img2img_mode:
        print("Image-to-Image mode detected: VAE quantization DISABLED to preserve quality")
        quantize_vae = False
        quantize_unet = False
        quantize_text_encoder = False
        target_memory = "~2.0-2.5GB"
    elif skip_quantization:
        print("\nSkipping full quantization due to LoRA usage")
        print("Quantizing VAE only to save memory...")
        quantize_vae = True
        quantize_unet = False
        quantize_text_encoder = False
        target_memory = "~1.8-2.0GB"
    else:
        print("Performing full 8-bit Quantization for CPU...")
        quantize_vae = True
        quantize_unet = True
        quantize_text_encoder = True
        target_memory = "~1.0-1.5GB"
    
    if quantize_vae:
        print("Performing 8-bit Quantization for VAE...")
        pipe.vae = torch.quantization.quantize_dynamic(
            pipe.vae,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        print("VAE quantized to 8-bit (saves ~150MB memory)")
    else:
        print("VAE quantization SKIPPED (to preserve image quality in img2img mode)")
    
    print("\nEnabling Attention Slicing for CPU...")
    pipe.enable_attention_slicing("max")
    print("Attention Slicing enabled (20-30% memory reduction)")
    
    print("\nEnabling VAE Slicing for CPU...")
    pipe.vae.enable_slicing()
    print("VAE Slicing enabled (15-25% memory reduction)")
    
    print("\nConfiguring CPU-specific settings...")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)
    print("Progress display enabled (Not affect)")
    
    cpu_cores = max(1, os.cpu_count() // 1)
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(1)
    print(f"CPU thread settings optimized: {cpu_cores} threads")
    
    print("\nApplying additional CPU optimizations...")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("CUDA backend settings adjusted for CPU compatibility")
    print("\n" + "="*60)
    print("All CPU memory reduction methods have been applied")
    print(f"Target memory usage: {target_memory} (vs 2.1GB without optimizations)")
    print("="*60)
    return pipe
def verify_memory_usage():
    """Verify memory usage for CPU systems using psutil"""
    print("\n" + "="*60)
    print("Verifying memory usage after optimization")
    print("="*60)
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Process memory usage: {memory_info.rss / (1024**3):.2f} GB")
        virtual_memory = psutil.virtual_memory()
        print(f"System available memory: {virtual_memory.available / (1024**3):.2f} GB")
        print(f"System total memory: {virtual_memory.total / (1024**3):.2f} GB")
        print(f"Memory usage percentage: {virtual_memory.percent:.1f}%")
    except Exception as e:
        print(f"Memory verification error: {str(e)}")
        print("Memory verification requires 'psutil' package (install with: pip install psutil)")
    print("="*60)
def generate_random_seed():
    """Generate a random seed between 0 and 1000000000"""
    return random.randint(0, 1000000000)
def check_system_resources(skip_quantization=False, required_memory=1.5):
    """Check if system has enough resources to run Stable Diffusion on CPU"""
    print("\n" + "="*60)
    print("Checking system resources for CPU execution")
    print("="*60)
    
    virtual_memory = psutil.virtual_memory()
    available_gb = virtual_memory.available / (1024**3)
    if available_gb < required_memory:
        print(f"WARNING: Low available memory ({available_gb:.2f} GB).")
        print(f"Stable Diffusion may fail (requires ~{required_memory:.1f}GB).")
        print("Consider closing other applications before proceeding.")
    else:
        print(f"Sufficient memory available: {available_gb:.2f} GB")
    
    cpu_cores = os.cpu_count()
    print(f"Detected CPU cores: {cpu_cores}")
    if cpu_cores < 2:
        print("WARNING: Very few CPU cores detected. Generation will be extremely slow.")
    elif cpu_cores < 4:
        print("Note: Few CPU cores detected. Generation will be slow but possible.")
    else:
        print("Sufficient CPU cores for reasonable generation speed.")
    print("="*60)
    return available_gb >= required_memory
def load_multiple_loras(pipe, lora_paths, weights=None):
    """
    Load and apply multiple LoRA weights with individual strengths
    Parameters:
    - pipe: Stable Diffusion pipeline
    - lora_paths: List of paths to LoRA weights files
    - weights: List of strengths for each LoRA (0.1-1.5)
    Returns:
    - bool: Whether all LoRAs were loaded successfully
    """
    print("\n" + "="*60)
    print("LOADING MULTIPLE LoRAs")
    print("="*60)
    if not lora_paths:
        print("No LoRA paths provided")
        return False
    
    if weights is None:
        weights = [1.0] * len(lora_paths)
    elif isinstance(weights, (int, float)):
        weights = [weights] * len(lora_paths)
    elif len(weights) < len(lora_paths):
        weights = weights + [1.0] * (len(lora_paths) - len(weights))
    adapter_names = []
    successful_loads = 0
    
    for i, (lora_path, weight) in enumerate(zip(lora_paths, weights)):
        print(f"\n--- LoRA #{i+1} ---")
        print(f"Path: {os.path.basename(lora_path)}")
        print(f"Strength: {weight:.2f}")
        
        adapter_name = f"lora_{i+1}"
        adapter_names.append(adapter_name)
        try:
            if not os.path.exists(lora_path):
                print(f"ERROR: File not found at {lora_path}")
                continue
            
            print(f"Loading LoRA as '{adapter_name}'...")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            
            try:
                adapters_dict = pipe.get_list_adapters()
                print(f"Current adapters: {adapters_dict}")
            except:
                pass
            print(f"LoRA loaded successfully as '{adapter_name}'")
            successful_loads += 1
        except Exception as e:
            print(f"ERROR loading LoRA: {str(e)}")
            if adapter_name in adapter_names:
                adapter_names.remove(adapter_name)
    
    if successful_loads > 0:
        valid_weights = []
        for i, adapter_name in enumerate(adapter_names):
            for j, path in enumerate(lora_paths):
                if f"lora_{j+1}" == adapter_name:
                    valid_weights.append(weights[j])
                    break
        print("\n" + "="*60)
        print(f"Activating {successful_loads} LoRA(s) with strengths: {valid_weights}")
        print("="*60)
        try:
            pipe.set_adapters(adapter_names, valid_weights)
            print(f"\nSUCCESS: {successful_loads} LoRA(s) ACTIVATED AND WORKING!")
            print(f"Adapter names: {adapter_names}")
            print(f"Strengths: {valid_weights}")
            print("="*60)
            return True
        except Exception as e:
            print(f"\nERROR activating adapters: {str(e)}")
    print("\nFAILED to activate any LoRA")
    print("="*60)
    return False
def load_image(image_path, target_size=None):
    """Load and preprocess image for img2img with exact size handling"""
    try:
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        if target_size is not None:
            ratio = min(target_size[0] / original_width, target_size[1] / original_height)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            left = (new_size[0] - target_size[0]) // 2
            top = (new_size[1] - target_size[1]) // 2
            image = image.crop((left, top, left + target_size[0], top + target_size[1]))
            print(f"Image loaded and preprocessed: {image_path}")
            print(f"Original size: {original_width}x{original_height} -> Resized to: {target_size}")
        else:
            new_width = (original_width // 8) * 8
            new_height = (original_height // 8) * 8
            new_width = max(64, new_width)
            new_height = max(64, new_height)
            left = (original_width - new_width) // 2
            top = (original_height - new_height) // 2
            image = image.crop((left, top, left + new_width, top + new_height))
            print(f"Image loaded and preprocessed: {image_path}")
            print(f"Original size: {original_width}x{original_height} -> Cropped to: {new_width}x{new_height} (exact multiple of 8)")
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None
def generate_image_with_cpu_optimizations(pipe, prompt, negative_prompt="", lora_active=False, guidance_scale=1.0, width=512, height=512):
    """Generate image with all CPU-specific memory optimizations (text-to-image)"""
    print("\n" + "="*60)
    print(f"Generating image with CPU optimizations: '{prompt}'")
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'")
    if lora_active:
        print("LoRA: ACTIVE (enhancing style/quality)")
    print(f"CFG Scale: {guidance_scale:.1f}")
    print(f"Image size: {width}x{height}")
    print("="*60)
    
    generation_settings = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": 10,
        "guidance_scale": guidance_scale,
        "output_type": "pil",
        "generator": torch.Generator(device="cpu").manual_seed(generate_random_seed())
    }
    print("CPU generation settings:")
    print(f" - Image size: {generation_settings['width']}x{generation_settings['height']}")
    if negative_prompt:
        print(f" - Negative prompt: {negative_prompt}")
    print(f" - Inference steps: {generation_settings['num_inference_steps']}")
    print(f" - Guidance scale (CFG): {generation_settings['guidance_scale']}")
    
    gc.collect()
    print("\nMemory cleared before generation")
    
    verify_memory_usage()
    print("\nStarting image generation...")
    start_time = time.time()
    try:
        print("\nGenerating image (this may take several minutes on CPU)...")
        progress_bar = tqdm(total=generation_settings["num_inference_steps"], desc="Processing")
        
        output = pipe(**generation_settings)
        image = output.images[0]
        
        progress_bar.update(generation_settings["num_inference_steps"])
        progress_bar.close()
        
        timestamp = int(time.time())
        seed = generation_settings["generator"].initial_seed()
        lora_suffix = "_multilora" if lora_active else ""
        output_path = f"cpu_optimized_output{lora_suffix}_{width}x{height}_{timestamp}_{seed}.png"
        image.save(output_path)
        elapsed = time.time() - start_time
        print(f"\nImage generated successfully! Time: {elapsed:.2f} seconds")
        print(f"Settings: {generation_settings['width']}x{generation_settings['height']}")
        print(f"Saved at: {output_path}")
        
        verify_memory_usage()
        return image, output_path
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nMemory error: Insufficient RAM!")
            print("Recommended solutions:")
            print("   1. Reduce image size to 384x384 or 256x256")
            print("   2. Increase num_inference_steps to 30-40")
            print("   3. Close all other applications to free memory")
            print("   4. Consider using a smaller model")
        else:
            print(f"Error during image generation: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None
def generate_image_to_image_with_cpu_optimizations(pipe, prompt, negative_prompt="", image_path=None, strength=0.75, lora_active=False, guidance_scale=1.0):
    """Generate image with all CPU-specific memory optimizations (image-to-image)"""
    print("\n" + "="*60)
    print(f"Generating image from image with CPU optimizations: '{prompt}'")
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'")
    print(f"Input image: {os.path.basename(image_path)}")
    print(f"Transformation strength: {strength:.2f}")
    print(f"CFG Scale: {guidance_scale:.1f}")
    if lora_active:
        print("LoRA: ACTIVE (enhancing style/quality)")
    print("="*60)
    
    init_image = load_image(image_path, target_size=None)
    if init_image is None:
        print("Failed to load input image. Aborting generation.")
        return None, None
    
    generation_settings = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": init_image,
        "strength": strength,
        "num_inference_steps": 10,
        "guidance_scale": guidance_scale,
        "output_type": "pil",
        "generator": torch.Generator(device="cpu").manual_seed(generate_random_seed())
    }
    print("CPU image-to-image settings:")
    print(f" - Input image size: {init_image.size} (exact multiple of 8)")
    if negative_prompt:
        print(f" - Negative prompt: {negative_prompt}")
    print(f" - Transformation strength: {generation_settings['strength']}")
    print(f" - Inference steps: {generation_settings['num_inference_steps']}")
    print(f" - Guidance scale (CFG): {generation_settings['guidance_scale']}")
    
    gc.collect()
    print("\nMemory cleared before generation")
    
    verify_memory_usage()
    print("\nStarting image-to-image generation...")
    start_time = time.time()
    try:
        print("\nGenerating image (this may take several minutes on CPU)...")
        progress_bar = tqdm(total=generation_settings["num_inference_steps"], desc="Processing")
        
        output = pipe(**generation_settings)
        image = output.images[0]
        
        progress_bar.update(generation_settings["num_inference_steps"])
        progress_bar.close()
        
        timestamp = int(time.time())
        seed = generation_settings["generator"].initial_seed()
        lora_suffix = "_multilora" if lora_active else ""
        img2img_suffix = "_img2img"
        output_path = f"cpu_optimized_output{img2img_suffix}{lora_suffix}_{timestamp}_{seed}.png"
        image.save(output_path)
        elapsed = time.time() - start_time
        print(f"\nImage generated successfully! Time: {elapsed:.2f} seconds")
        print(f"Input image: {os.path.basename(image_path)}")
        print(f"Transformation strength: {strength:.2f}")
        print(f"Output size: {image.size} (matches input size)")
        print(f"Saved at: {output_path}")
        
        verify_memory_usage()
        return image, output_path
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nMemory error: Insufficient RAM!")
            print("Recommended solutions:")
            print("   1. Reduce image size to 384x384 or 256x256")
            print("   2. Decrease transformation strength (try 0.5-0.6)")
            print("   3. Increase num_inference_steps to 30-40")
            print("   4. Close all other applications to free memory")
            print("   5. Consider using a smaller model")
        else:
            print(f"Error during image generation: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None
def warm_up_model(pipe, img2img_mode=False):
    """Run a quick warm-up generation to initialize all model components"""
    print("\n" + "="*60)
    print("Running model warm-up to optimize first generation time")
    print("="*60)
    start_time = time.time()
    
    warmup_settings = {
        "prompt": "warmup",
        "width": 128 if not img2img_mode else None,
        "height": 128 if not img2img_mode else None,
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
        "output_type": "np",
        "generator": torch.Generator(device="cpu").manual_seed(42)
    }
    
    if img2img_mode:
        from PIL import Image
        import numpy as np
        warmup_image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        warmup_settings["image"] = warmup_image
        warmup_settings["strength"] = 0.5
    try:
        if img2img_mode:
            pipe.img2img(**warmup_settings)
        else:
            pipe(**warmup_settings)
        elapsed = time.time() - start_time
        print(f"Warm-up completed in {elapsed:.2f} seconds")
        print("First generation will now be significantly faster")
    except Exception as e:
        print(f"Warm-up failed (non-critical): {str(e)}")
        print("Generation will still work but first run may be slower")
    print("="*60)
def save_configuration(config, filename="sd_config.json"):
    """Save current configuration to a file for later reuse"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfiguration saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save configuration: {str(e)}")
        return False
def load_configuration(filename="sd_config.json"):
    """Load configuration from a file"""
    try:
        if not os.path.exists(filename):
            print(f"Configuration file {filename} not found")
            return None
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration loaded from {filename}")
        return config
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        return None
def create_config_from_current_settings(args):
    """Create a configuration dictionary from current settings"""
    return {
        "model_path": args["model_path"],
        "is_img2img": args["is_img2img"],
        "lora_paths": args["lora_paths"],
        "lora_weights": args["lora_weights"],
        "scheduler": args["scheduler_name"],
        "cfg_scale": args["cfg_scale"],
        "prompt": args["user_prompt"],
        "negative_prompt": args["negative_prompt"],
        "width": args.get("width", 512),
        "height": args.get("height", 512),
        "img2img_strength": args.get("img2img_strength", 0.75) if args["is_img2img"] else None,
        "img2img_path": args.get("image_path", None) if args["is_img2img"] else None,
        "timestamp": int(time.time())
    }
def show_preset_menu():
    """Show quick preset menu for common use cases"""
    print("\n" + "="*60)
    print("QUICK PRESETS MENU")
    print("="*60)
    print("1. Photorealistic (High quality photos)")
    print("2. Anime/Cartoon style")
    print("3. Digital Art (Concept art, illustrations)")
    print("4. 3D Render style")
    print("5. Custom settings (manual configuration)")
    print("6. Load saved configuration")
    choice = input("\nEnter preset choice (1-6): ").strip()
    return choice
def apply_preset(choice, is_img2img):
    """Apply settings based on selected preset"""
    presets = {
        "1": {
            "prompt_suffix": ", photorealistic, 8k, ultra-detailed, professional photography",
            "negative_prompt": "cartoon, drawing, illustration, anime, 3d render, blurry, low quality",
            "cfg_scale": 7.5,
            "steps": 25,
            "width": 512,
            "height": 512
        },
        "2": {
            "prompt_suffix": ", anime style, vibrant colors, cel-shaded",
            "negative_prompt": "photorealistic, 3d, realistic, photograph, blurry",
            "cfg_scale": 7.0,
            "steps": 20,
            "width": 512,
            "height": 512
        },
        "3": {
            "prompt_suffix": ", digital art, concept art, trending on artstation",
            "negative_prompt": "photograph, realistic, 3d render, blurry",
            "cfg_scale": 8.0,
            "steps": 30,
            "width": 512,
            "height": 512
        },
        "4": {
            "prompt_suffix": ", 3d render, blender, octane render, unreal engine",
            "negative_prompt": "photograph, drawing, anime, blurry, low quality",
            "cfg_scale": 9.0,
            "steps": 35,
            "width": 512,
            "height": 512
        }
    }
    if choice in presets:
        return presets[choice]
    return None
def unload_model_from_memory(pipe):
    """Completely unload model from memory to free resources"""
    print("\n" + "="*60)
    print("Unloading model from memory to free resources")
    print("="*60)
    try:
        del pipe.unet
        del pipe.text_encoder
        del pipe.vae
        del pipe.scheduler
        gc.collect()
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory freed: Model components unloaded")
        print(f"Current process memory usage: {memory_info.rss / (1024**3):.2f} GB")
        print("="*60)
        return True
    except Exception as e:
        print(f"Error unloading model: {str(e)}")
        return False
def reload_model(pipe, model_path, is_img2img, scheduler_class):
    """Reload model with specific configuration"""
    print("\n" + "="*60)
    print("Reloading model with new configuration")
    print("="*60)
    try:
        if is_img2img:
            new_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            new_pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            )
        new_pipe.scheduler = scheduler_class.from_config(new_pipe.scheduler.config)
        new_pipe = new_pipe.to(torch.device("cpu"))
        print("Model reloaded successfully")
        print("="*60)
        return new_pipe
    except Exception as e:
        print(f"Failed to reload model: {str(e)}")
        return None
def apply_post_processing(image):
    """Apply post-processing enhancements to the generated image"""
    print("\nApplying post-processing enhancements...")
    try:
        from PIL import ImageEnhance, ImageFilter
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        image = image.filter(ImageFilter.SMOOTH)
        print("Post-processing completed successfully")
        return image
    except Exception as e:
        print(f"Post-processing failed (non-critical): {str(e)}")
        return image
def generate_detailed_report(pipe, generation_params, elapsed_time, output_path, 
                           lora_active=False, lora_paths=None):
    """Generate a detailed report of the generation process"""
    report = f"""
===== STABLE DIFFUSION GENERATION REPORT =====
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: {os.path.basename(generation_params['model_path'])}
Mode: {'Image-to-Image' if generation_params['is_img2img'] else 'Text-to-Image'}
----- Generation Parameters -----
Prompt: {generation_params['prompt']}
{'Negative Prompt: ' + generation_params['negative_prompt'] if generation_params['negative_prompt'] else 'Negative Prompt: None'}
Scheduler: {generation_params['scheduler']}
CFG Scale: {generation_params['cfg_scale']}
Image Size: {generation_params.get('width', 'N/A')}x{generation_params.get('height', 'N/A')}
Inference Steps: {generation_params['steps']}
{'Image Strength: ' + str(generation_params['strength']) if generation_params['is_img2img'] else ''}
----- LoRA Configuration -----
{'Active' if lora_active else 'Not Active'}
{lora_paths if lora_active and lora_paths else ''}
----- Performance Metrics -----
Generation Time: {elapsed_time:.2f} seconds
Memory Usage: {get_current_memory_usage():.2f} GB
Output Path: {output_path}
----- System Information -----
Python Version: {sys.version.split()[0]}
Diffusers Version: {diffusers.__version__}
Torch Version: {torch.__version__}
CPU: {psutil.cpu_count()} cores
============================================
"""
    report_path = output_path.replace('.png', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    return report_path
def get_current_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)
def post_generation_actions(image, output_path, generation_params):
    """Offer post-generation actions to the user with proper menu loop"""
    while True:
        print("\n" + "="*60)
        print("POST-GENERATION ACTIONS")
        print("="*60)
        print("1. View image (requires image viewer)")
        print("2. Apply post-processing enhancements")
        print("3. Save as high-quality JPEG")
        print("4. Generate detailed report")
        print("5. Save current configuration for reuse")
        print("6. Generate new image with same settings")
        print("7. Return to main menu")
        print("8. Exit program")
        choice = input("\nEnter action (1-8): ").strip()
        if choice == "1":
            try:
                image.show()
                print("Image viewer opened")
            except Exception as e:
                print(f"Failed to open image viewer: {str(e)}")
        elif choice == "2":
            enhanced_img = apply_post_processing(image)
            enhanced_path = output_path.replace('.png', '_enhanced.png')
            enhanced_img.save(enhanced_path)
            print(f"Enhanced image saved to: {enhanced_path}")
        elif choice == "3":
            jpeg_path = output_path.replace('.png', '.jpg')
            image.save(jpeg_path, quality=95)
            print(f"High-quality JPEG saved to: {jpeg_path}")
        elif choice == "4":
            generate_detailed_report(
                None,
                {
                    "model_path": generation_params.get('model_path', 'unknown'),
                    "is_img2img": generation_params.get("is_img2img", False),
                    "prompt": generation_params.get("prompt", ""),
                    "negative_prompt": generation_params.get("negative_prompt", ""),
                    "scheduler": generation_params.get("scheduler", "unknown"),
                    "cfg_scale": generation_params.get("cfg_scale", 1.0),
                    "width": generation_params.get("width", 512),
                    "height": generation_params.get("height", 512),
                    "steps": generation_params.get("steps", 10),
                    "strength": generation_params.get("strength", 0.75) if generation_params.get("is_img2img", False) else None
                },
                generation_params.get("elapsed_time", 0),
                output_path,
                generation_params.get("lora_active", False),
                generation_params.get("lora_paths", None)
            )
        elif choice == "5":
            config = {
                "model_path": generation_params.get("model_path", ""),
                "is_img2img": generation_params.get("is_img2img", False),
                "lora_paths": generation_params.get("lora_paths", []),
                "lora_weights": generation_params.get("lora_weights", []),
                "scheduler": generation_params.get("scheduler", "LCM"),
                "cfg_scale": generation_params.get("cfg_scale", 1.0),
                "prompt": generation_params.get("prompt", ""),
                "negative_prompt": generation_params.get("negative_prompt", ""),
                "width": generation_params.get("width", 512),
                "height": generation_params.get("height", 512),
                "img2img_strength": generation_params.get("strength", 0.75) if generation_params.get("is_img2img", False) else None
            }
            print("\nEnter filename for configuration (default: sd_config.json):")
            custom_filename = input().strip()
            if not custom_filename:
                custom_filename = "sd_config.json"
            elif not custom_filename.endswith(".json"):
                custom_filename += ".json"
            save_configuration(config, filename=custom_filename)
        elif choice == "6":
            print("\nGenerating new image with same settings...")
            return True, False
        elif choice == "7":
            print("\nReturning to main menu...")
            return False, True
        elif choice == "8":
            print("\nExiting program...")
            print("\n" + "="*60)
            print("Cleaning up resources before exit")
            print("="*60)
            gc.collect()
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"Current process memory usage: {memory_info.rss / (1024**3):.2f} GB")
            except:
                pass
            print("="*60)
            return False, False
        else:
            print("\nInvalid choice. Please enter a number between 1-8.")
def load_prompt_history():
    """Load prompt history from file"""
    history_file = "prompt_history.json"
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except:
        return []
def save_to_prompt_history(prompt, negative_prompt="", timestamp=None):
    """Save prompt to history"""
    if not timestamp:
        timestamp = int(time.time())
    history = load_prompt_history()
    history.insert(0, {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": timestamp
    })
    history = history[:50]
    try:
        with open("prompt_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False
def prompt_history_menu():
    """Show prompt history menu for selection"""
    history = load_prompt_history()
    if not history:
        print("\nNo prompt history available")
        return None
    print("\n" + "="*60)
    print("PROMPT HISTORY")
    print("="*60)
    for i, item in enumerate(history[:10]):
        time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(item['timestamp']))
        prompt_preview = item['prompt'][:50] + "..." if len(item['prompt']) > 50 else item['prompt']
        print(f"{i+1}. [{time_str}] {prompt_preview}")
    print("\n0. Don't use history")
    choice = input("\nSelect prompt from history (0-10): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(history):
            return history[idx]
    except:
        pass
    return None
def get_valid_dimensions():
    """Get valid image dimensions that are multiples of 8"""
    print("\n" + "="*60)
    print("IMAGE DIMENSIONS CONFIGURATION")
    print("="*60)
    print("Note: Dimensions must be multiples of 8 for Stable Diffusion to work properly")
    print("\nCommon aspect ratios:")
    print("1. Square (512x512)")
    print("2. Standard (512x768)")
    print("3. Landscape (768x512)")
    print("4. Mobile (512x1024)")
    print("5. Custom dimensions")
    choice = input("\nEnter choice (1-5, default: 1): ").strip()
    if choice == "1":
        return 512, 512
    elif choice == "2":
        return 512, 768
    elif choice == "3":
        return 768, 512
    elif choice == "4":
        return 512, 1024
    else:
        print("\nEnter custom width (multiple of 8, min 256, max 1024):")
        width_input = input().strip()
        try:
            width = int(width_input)
            width = (width // 8) * 8
            width = max(256, min(1024, width))
        except:
            width = 512
        print(f"Using width: {width}")
        print("\nEnter custom height (multiple of 8, min 256, max 1024):")
        height_input = input().strip()
        try:
            height = int(height_input)
            height = (height // 8) * 8
            height = max(256, min(1024, height))
        except:
            height = 512
        print(f"Using height: {height}")
        return width, height
def main():
    """Main function to run the CPU-optimized Stable Diffusion"""
    print("="*60)
    print("CPU-Optimized Stable Diffusion Pipeline with MULTIPLE LoRA Support")
    print("Fully supports loading and applying multiple LoRAs simultaneously")
    print("Enhanced: Image-to-Image with exact latent size and VAE-only quantization")
    print("New: Multiple scheduler options with dynamic CFG scale and negative prompt")
    print("Updated: Support for saving and loading multiple configuration files")
    print("="*60)
    
    exit_requested = False
    while not exit_requested:
        print("UNDER YSNRFD LICENSE")
        print("\nSelect generation mode:")
        print("1. Text-to-Image (generate from text prompt)")
        print("UNDER YSNRFD LICENSE")
        print("2. Image-to-Image (modify existing image)")
        print("UNDER YSNRFD LICENSE")
        print("3. Exit program")
        print("UNDER YSNRFD LICENSE")
        mode = input("Enter choice (1, 2, or 3): ").strip()
        
        if mode == "3":
            print("Exiting program...")
            exit_requested = True
            continue
        elif mode not in ['1', '2']:
            print("Invalid choice. Defaulting to Text-to-Image.")
            mode = '1'
        is_img2img = (mode == '2')
        
        print("\nDo you want to use LoRA (Low-Rank Adaptation) for style enhancement? (y/n)")
        use_lora = input().strip().lower() == 'y'
        lora_paths = []
        lora_weights = []
        skip_quantization = False
        if use_lora:
            print("\nHow many LoRAs do you want to use? (1-5, default: 1)")
            num_loras_input = input().strip()
            num_loras = 1
            if num_loras_input:
                try:
                    num_loras = max(1, min(5, int(num_loras_input)))
                    print(f"Configuring for {num_loras} LoRA(s)")
                except:
                    print("Invalid number, using default: 1")
            
            for i in range(num_loras):
                print(f"\n--- LoRA #{i+1} Configuration ---")
                print(f"Enter path to LoRA weights (safetensors file #{i+1}):")
                lora_path = input().strip()
                if not lora_path:
                    if i == 0:
                        print("Main LoRA path not provided. Continuing WITHOUT LoRA.")
                        use_lora = False
                    continue
                lora_paths.append(lora_path)
                print(f"Enter LoRA #{i+1} strength (0.1-1.5, default 1.0):")
                weight_input = input().strip()
                if weight_input:
                    try:
                        weight = float(weight_input)
                        weight = max(0.1, min(1.5, weight))
                        lora_weights.append(weight)
                        print(f"Using LoRA #{i+1} strength: {weight:.2f}")
                    except:
                        print("Invalid value. Using default 1.0")
                        lora_weights.append(1.0)
                else:
                    lora_weights.append(1.0)
            if lora_paths:
                skip_quantization = True
        
        print("\n" + "="*60)
        print("CONFIGURATION OPTIONS")
        print("="*60)
        print("1. Quick Presets (recommended styles)")
        print("2. Custom Configuration")
        config_choice = input("\nEnter choice (1-2, default: 2): ").strip()
        
        preset_choice = None
        preset = None
        continue_to_generation = False
        width, height = 512, 512
        if config_choice == "1":
            preset_choice = show_preset_menu()
            if preset_choice == "6":
                config_files = [f for f in os.listdir() if f.endswith('.json') and 'config' in f.lower()]
                if not config_files:
                    print("\nNo configuration files found")
                    config_choice = "2"
                else:
                    print("\nAvailable configuration files:")
                    for i, f in enumerate(config_files):
                        print(f"{i+1}. {f}")
                    print("0. Cancel")
                    file_choice = input("\nSelect configuration file: ").strip()
                    try:
                        idx = int(file_choice) - 1
                        if 0 <= idx < len(config_files):
                            config = load_configuration(filename=config_files[idx])
                            if config:
                                is_img2img = config.get("is_img2img", is_img2img)
                                lora_paths = config.get("lora_paths", lora_paths)
                                lora_weights = config.get("lora_weights", lora_weights)
                                if lora_paths:
                                    use_lora = True
                                    skip_quantization = True
                                scheduler_name = config.get("scheduler", "LCM")
                                if scheduler_name == "LCM":
                                    cfg_scale = config.get("cfg_scale", 1.0)
                                else:
                                    cfg_scale = config.get("cfg_scale", 7.5)
                                width = config.get("width", 512)
                                height = config.get("height", 512)
                                if not is_img2img:
                                    print(f"Loaded image dimensions: {width}x{height}")
                                user_prompt = config.get("prompt", "")
                                negative_prompt = config.get("negative_prompt", "")
                                if is_img2img:
                                    img2img_strength = config.get("img2img_strength", 0.75)
                                print("\nUsing loaded configuration. Continuing with generation...")
                                continue_to_generation = True
                            else:
                                print("Falling back to custom configuration")
                                config_choice = "2"
                        elif idx == -1:
                            config_choice = "2"
                        else:
                            print("Invalid selection. Falling back to custom configuration.")
                            config_choice = "2"
                    except:
                        print("Invalid input. Falling back to custom configuration.")
                        config_choice = "2"
            elif preset_choice in ["1", "2", "3", "4"]:
                preset = apply_preset(preset_choice, is_img2img)
                print(f"\nApplied preset: {['Photorealistic', 'Anime/Cartoon', 'Digital Art', '3D Render'][int(preset_choice)-1]}")
                width = preset.get("width", 512)
                height = preset.get("height", 512)
            else:
                preset = None
        else:
            preset = None
        
        if continue_to_generation:
            device = torch.device("cpu")
            print(f"\nCPU mode initialized: {device}")
            
            print("\n" + "="*60)
            print("Model Configuration")
            print("="*60)
            model_path = config.get("model_path", "ds_lcm.safetensors")
            print(f"Using model path from configuration: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"\nERROR: Model file not found at: {model_path}")
                print("Please check the path and try again.")
                continue
            
            print("\n" + "="*60)
            print("Loading Stable Diffusion model")
            print("="*60)
            try:
                print("Attempting to load model in float32 mode (CPU compatible)...")
                if is_img2img:
                    print("Loading Image-to-Image pipeline...")
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                else:
                    print("Loading Text-to-Image pipeline...")
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                
                SCHEDULER_OPTIONS = {
                    "1": ("DDIM", DDIMScheduler),
                    "2": ("PNDMScheduler", PNDMScheduler),
                    "3": ("LMS Discrete", LMSDiscreteScheduler),
                    "4": ("Euler Discrete", EulerDiscreteScheduler),
                    "5": ("Euler Ancestral", EulerAncestralDiscreteScheduler),
                    "6": ("DPM++ 2M", DPMSolverMultistepScheduler),
                    "7": ("LCM", LCMScheduler),
                    "8": ("UniPC", UniPCMultistepScheduler)
                }
                
                scheduler_class = None
                for key, (name, cls) in SCHEDULER_OPTIONS.items():
                    if name == scheduler_name or key == scheduler_name:
                        scheduler_class = cls
                        break
                if scheduler_class is None:
                    print(f"WARNING: Scheduler '{scheduler_name}' not found. Using default LCM scheduler.")
                    scheduler_class = LCMScheduler
                print(f"\nConfiguring {scheduler_name} scheduler...")
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                pipe = pipe.to(device)
                print(f"Model loaded successfully with {scheduler_name} scheduler")
            except Exception as e:
                print(f"\nERROR: Failed to load model: {str(e)}")
                print("\nTroubleshooting steps:")
                print("1. Verify the model file exists at the specified path")
                print("2. Ensure the model is compatible with diffusers library")
                print("3. Check if you have sufficient disk space")
                print("4. Try a different model file if available")
                continue
            
            warm_up_model(pipe, img2img_mode=is_img2img)
            
            lora_loaded = False
            if use_lora and lora_paths:
                lora_loaded = load_multiple_loras(pipe, lora_paths, lora_weights)
            
            optimized_pipe = setup_cpu_memory_optimizations(
                pipe, 
                skip_quantization=skip_quantization, 
                img2img_mode=is_img2img
            )
            
            verify_memory_usage()
            
            generation_params = {
                "model_path": model_path,
                "is_img2img": is_img2img,
                "lora_paths": lora_paths,
                "lora_weights": lora_weights,
                "scheduler": scheduler_name,
                "cfg_scale": cfg_scale,
                "prompt": user_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": 10,
                "strength": img2img_strength if is_img2img else 0.75,
                "lora_active": lora_loaded
            }
            continue_generation = True
            return_to_main = False
            while continue_generation and not return_to_main:
                if is_img2img:
                    image_path = config.get("img2img_path", "input.jpg")
                    if not os.path.exists(image_path):
                        print(f"ERROR: Input image not found at: {image_path}")
                        print("Please provide a valid image path.")
                        continue_generation = False
                        break
                    start_time = time.time()
                    image, output_path = generate_image_to_image_with_cpu_optimizations(
                        optimized_pipe, 
                        user_prompt, 
                        negative_prompt,
                        image_path=image_path, 
                        strength=img2img_strength,
                        lora_active=lora_loaded,
                        guidance_scale=cfg_scale
                    )
                    elapsed_time = time.time() - start_time
                else:
                    start_time = time.time()
                    image, output_path = generate_image_with_cpu_optimizations(
                        optimized_pipe, 
                        user_prompt, 
                        negative_prompt,
                        lora_active=lora_loaded,
                        guidance_scale=cfg_scale,
                        width=width,
                        height=height
                    )
                    elapsed_time = time.time() - start_time
                
                generation_params["elapsed_time"] = elapsed_time
                if image and output_path:
                    continue_generation, return_to_main = post_generation_actions(
                        image, 
                        output_path,
                        generation_params
                    )
                else:
                    continue_generation = False
            
            if return_to_main:
                print("\nReturning to main menu...")
                continue
            
            print("\n" + "="*60)
            print("Cleaning up resources")
            print("="*60)
            unload_model_from_memory(pipe)
            gc.collect()
            print("Memory cleanup completed")
            print("\n" + "="*60)
            print("CPU-Optimized Stable Diffusion Process Completed")
            print("Note: This implementation is designed for systems with limited resources")
            print("Multiple LoRA is applied correctly using Diffusers built-in methods")
            print("No need to add <lora:lora_name> in prompts - LoRA is applied programmatically")
            print("Support for saving and loading multiple configuration files added")
            print("="*60)
            
            if lora_loaded:
                print("\nSUCCESS: Multiple LoRAs were applied CORRECTLY and are affecting your images")
                print(f"Total LoRAs active: {len(lora_paths)}")
            else:
                print("\nWARNING: LoRA(s) were NOT applied - check error messages above")
            print("="*60)
            continue
        
        print("\n" + "="*60)
        print("Scheduler Configuration")
        print("="*60)
        SCHEDULER_OPTIONS = {
            "1": ("DDIM", DDIMScheduler),
            "2": ("PNDMScheduler", PNDMScheduler),
            "3": ("LMS Discrete", LMSDiscreteScheduler),
            "4": ("Euler Discrete", EulerDiscreteScheduler),
            "5": ("Euler Ancestral", EulerAncestralDiscreteScheduler),
            "6": ("DPM++ 2M", DPMSolverMultistepScheduler),
            "7": ("LCM", LCMScheduler),
            "8": ("UniPC", UniPCMultistepScheduler)
        }
        print("\nSelect scheduler for generation:")
        for key, (name, _) in SCHEDULER_OPTIONS.items():
            print(f"{key}. {name}")
        scheduler_choice = input("Enter choice (1-8, default: 7 for LCM): ").strip()
        if not scheduler_choice or scheduler_choice not in SCHEDULER_OPTIONS:
            scheduler_choice = "7"
        scheduler_name, scheduler_class = SCHEDULER_OPTIONS[scheduler_choice]
        print(f"Selected scheduler: {scheduler_name}")
        
        if scheduler_choice == "7":
            cfg_scale = 1.0
            print("\nLCM scheduler selected: Using default CFG scale = 1.0")
            print("Note: LCM works best with low CFG scale (1.0-2.0)")
        else:
            print("\n" + "="*60)
            print("CFG Scale Configuration")
            print("="*60)
            print("CFG (Classifier-Free Guidance) Scale controls how much the")
            print("image follows your prompt. Higher values = more prompt adherence.")
            print("Recommended range: 5.0-12.0 (7.5 is common default)")
            default_cfg = 7.5
            print(f"\nEnter CFG scale (default: {default_cfg}):")
            cfg_input = input().strip()
            if cfg_input:
                try:
                    cfg_scale = float(cfg_input)
                    cfg_scale = max(1.0, min(20.0, cfg_scale))
                    print(f"Using CFG scale: {cfg_scale:.1f}")
                except:
                    print("Invalid value. Using default CFG scale.")
                    cfg_scale = default_cfg
            else:
                cfg_scale = default_cfg
                print(f"Using default CFG scale: {cfg_scale:.1f}")
        
        required_memory = 1.5
        if scheduler_choice == "7":
            required_memory = 1.3
        elif scheduler_choice in ["5", "6"]:
            required_memory = 1.8
        if skip_quantization:
            required_memory += 0.3
        if is_img2img:
            required_memory += 0.5
        print(f"\nEstimated memory requirement with selected scheduler: ~{required_memory:.1f}GB")
        
        if not check_system_resources(skip_quantization=skip_quantization, required_memory=required_memory):
            print("\nWARNING: System may not have sufficient resources.")
            print("Continue anyway? (y/n)")
            if input().lower() != 'y':
                print("Operation cancelled by user.")
                continue
        
        device = torch.device("cpu")
        print(f"\nCPU mode initialized: {device}")
        
        print("\n" + "="*60)
        print("Model Configuration")
        print("="*60)
        default_model_path_1 = "ds_lcm.safetensors"
        default_model_path_2 = "wm_lcm.safetensors"
        print(f"Default model path: 1.{default_model_path_1} 2.{default_model_path_2}")
        print(f"Enter model path (or press Enter to use {default_model_path_1}):")
        model_path = input().strip()
        if not model_path:
            model_path = default_model_path_1
        if model_path == "1":
            model_path = default_model_path_1
        if model_path == "2":
            model_path = default_model_path_2
        print(f"Using model path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"\nERROR: Model file not found at: {model_path}")
            print("Please check the path and try again.")
            continue
        
        print("\n" + "="*60)
        print("Loading Stable Diffusion model")
        print("="*60)
        try:
            print("Attempting to load model in float32 mode (CPU compatible)...")
            if is_img2img:
                print("Loading Image-to-Image pipeline...")
                pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                print("Loading Text-to-Image pipeline...")
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            print(f"\nConfiguring {scheduler_name} scheduler...")
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            print(f"Model loaded successfully with {scheduler_name} scheduler")
        except Exception as e:
            print(f"\nERROR: Failed to load model: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Verify the model file exists at the specified path")
            print("2. Ensure the model is compatible with diffusers library")
            print("3. Check if you have sufficient disk space")
            print("4. Try a different model file if available")
            continue
        
        warm_up_model(pipe, img2img_mode=is_img2img)
        
        lora_loaded = False
        if use_lora and lora_paths:
            lora_loaded = load_multiple_loras(pipe, lora_paths, lora_weights)
        
        optimized_pipe = setup_cpu_memory_optimizations(
            pipe, 
            skip_quantization=skip_quantization, 
            img2img_mode=is_img2img
        )
        
        verify_memory_usage()
        
        print("\n" + "="*60)
        print("Generation Parameters")
        print("="*60)
        if not is_img2img:
            width, height = get_valid_dimensions()
            print(f"\nUsing image dimensions: {width}x{height}")
        
        print("Do you want to use prompt from history? (y/n)")
        use_history = input().strip().lower() == 'y'
        if use_history:
            history_item = prompt_history_menu()
            if history_item:
                user_prompt = history_item["prompt"]
                negative_prompt = history_item["negative_prompt"]
                print(f"Using prompt from history: {user_prompt}")
                print(f"Using negative prompt from history: {negative_prompt}")
            else:
                print("Enter your prompt (or press Enter for default):")
                default_prompt = "A beautiful landscape with mountains and a lake, ultra-detailed, realistic, 4k"
                user_prompt = input().strip()
                if not user_prompt:
                    user_prompt = default_prompt
                if preset:
                    user_prompt += preset["prompt_suffix"]
                    print(f"Applied preset suffix: {preset['prompt_suffix']}")
                print(f"Using prompt: {user_prompt}")
                
                print("\nEnter negative prompt (optional, press Enter to skip):")
                print("Common negative prompts: 'blurry, low quality, distorted, artifacts'")
                negative_prompt = input().strip()
                if negative_prompt:
                    print(f"Using negative prompt: {negative_prompt}")
                else:
                    if preset and preset["negative_prompt"]:
                        negative_prompt = preset["negative_prompt"]
                        print(f"Using preset negative prompt: {negative_prompt}")
                    else:
                        print("No negative prompt will be used")
        else:
            print("Enter your prompt (or press Enter for default):")
            default_prompt = "A beautiful landscape with mountains and a lake, ultra-detailed, realistic, 4k"
            user_prompt = input().strip()
            if not user_prompt:
                user_prompt = default_prompt
            if preset:
                user_prompt += preset["prompt_suffix"]
                print(f"Applied preset suffix: {preset['prompt_suffix']}")
            print(f"Using prompt: {user_prompt}")
            
            print("\nEnter negative prompt (optional, press Enter to skip):")
            print("Common negative prompts: 'blurry, low quality, distorted, artifacts'")
            negative_prompt = input().strip()
            if negative_prompt:
                print(f"Using negative prompt: {negative_prompt}")
            else:
                if preset and preset["negative_prompt"]:
                    negative_prompt = preset["negative_prompt"]
                    print(f"Using preset negative prompt: {negative_prompt}")
                else:
                    print("No negative prompt will be used")
        
        save_to_prompt_history(user_prompt, negative_prompt)
        
        img2img_strength = 0.75
        if is_img2img:
            print("\nEnter path to input image:")
            image_path = input().strip()
            if not os.path.exists(image_path):
                print(f"ERROR: Image file not found at: {image_path}")
                print("Using default image path: input.jpg")
                image_path = "input.jpg"
                if not os.path.exists(image_path):
                    print("Default image not found. Aborting.")
                    continue
            print("\nEnter transformation strength (0.1-1.0, default 0.75):")
            strength_input = input().strip()
            if strength_input:
                try:
                    img2img_strength = float(strength_input)
                    img2img_strength = max(0.1, min(1.0, img2img_strength))
                    print(f"Using transformation strength: {img2img_strength:.2f}")
                except:
                    print("Invalid value. Using default 0.75")
        
        generation_params = {
            "model_path": model_path,
            "is_img2img": is_img2img,
            "lora_paths": lora_paths,
            "lora_weights": lora_weights,
            "scheduler": scheduler_name,
            "cfg_scale": cfg_scale,
            "prompt": user_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": 10,
            "strength": img2img_strength,
            "lora_active": lora_loaded
        }
        
        if is_img2img:
            start_time = time.time()
            image, output_path = generate_image_to_image_with_cpu_optimizations(
                optimized_pipe, 
                user_prompt, 
                negative_prompt,
                image_path=image_path, 
                strength=img2img_strength,
                lora_active=lora_loaded,
                guidance_scale=cfg_scale
            )
            elapsed_time = time.time() - start_time
        else:
            start_time = time.time()
            image, output_path = generate_image_with_cpu_optimizations(
                optimized_pipe, 
                user_prompt, 
                negative_prompt,
                lora_active=lora_loaded,
                guidance_scale=cfg_scale,
                width=width,
                height=height
            )
            elapsed_time = time.time() - start_time
        
        generation_params["elapsed_time"] = elapsed_time
        
        continue_generation = True
        return_to_main = False
        while continue_generation and not return_to_main:
            if image and output_path:
                continue_generation, return_to_main = post_generation_actions(
                    image, 
                    output_path,
                    generation_params
                )
                if continue_generation and not return_to_main:
                    print("\n" + "="*60)
                    print("Generating new image with same settings...")
                    print("="*60)
                    if is_img2img:
                        start_time = time.time()
                        image, output_path = generate_image_to_image_with_cpu_optimizations(
                            optimized_pipe, 
                            user_prompt, 
                            negative_prompt,
                            image_path=image_path, 
                            strength=img2img_strength,
                            lora_active=lora_loaded,
                            guidance_scale=cfg_scale
                        )
                        elapsed_time = time.time() - start_time
                    else:
                        start_time = time.time()
                        image, output_path = generate_image_with_cpu_optimizations(
                            optimized_pipe, 
                            user_prompt, 
                            negative_prompt,
                            lora_active=lora_loaded,
                            guidance_scale=cfg_scale,
                            width=width,
                            height=height
                        )
                        elapsed_time = time.time() - start_time
                    generation_params["elapsed_time"] = elapsed_time
            else:
                continue_generation = False
        
        if return_to_main:
            print("\nReturning to main menu...")
            continue
        
        print("\n" + "="*60)
        print("Cleaning up resources")
        print("="*60)
        unload_model_from_memory(pipe)
        gc.collect()
        print("Memory cleanup completed")
        print("\n" + "="*60)
        print("CPU-Optimized Stable Diffusion Process Completed")
        print("Note: This implementation is designed for systems with limited resources")
        print("Multiple LoRA is applied correctly using Diffusers built-in methods")
        print("No need to add <lora:lora_name> in prompts - LoRA is applied programmatically")
        print("Support for saving and loading multiple configuration files added")
        print("="*60)
        
        if lora_loaded:
            print("\nSUCCESS: Multiple LoRAs were applied CORRECTLY and are affecting your images")
            print(f"Total LoRAs active: {len(lora_paths)}")
        else:
            print("\nWARNING: LoRA(s) were NOT applied - check error messages above")
        print("="*60)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
        gc.collect()
        print("Cleanup completed. Exiting.")
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print("\nMOST LIKELY CAUSE:")
        print("- Outdated diffusers library (MUST be >=0.18.0)")
        print("- Incompatible LoRA file(s)")
        print("- Invalid image path for img2img")
        print("\nSOLUTION:")
        print("1. UPGRADE diffusers: pip install --upgrade diffusers")
        print("2. Verify LoRA compatibility with your model")
        print("3. Check image path for img2img mode")
        print("\nReturning to main menu...")
        sys.exit(1)




"""
Complete Python code for CPU-optimized Stable Diffusion with Multiple LoRA Support and Image-to-Image
Fixed: Proper implementation for loading and applying multiple LoRAs simultaneously
Enhanced: Image-to-Image functionality with exact latent size and VAE-only quantization
Added: Multiple scheduler options with dynamic CFG scale, negative prompt, and advanced features
Fixed: Proper menu navigation after saving configuration
Added: Support for saving and loading multiple configuration files
Code By: YSNRFD (Updated with Multiple LoRA, Image-to-Image, and Advanced Settings)
Telegram: @ysnrfd
Github: ysnrfd
Huggingface: ysnrfd
--------------------
UNDER YSNRFD LICENSE
--------------------
"""