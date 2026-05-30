"""
CPU-Optimized Stable Diffusion Pipeline — SD 1.5 & SDXL Edition
================================================================
Complete support for:
    Stable Diffusion 1.5 (Text-to-Image, Image-to-Image)
    Stable Diffusion XL   (Text-to-Image, Image-to-Image, Refiner)
    Multiple LoRA with individual strengths
    9 scheduler options with auto CFG
    Batch generation, seed control, prompt history
    Post-processing, config save/load, presets
    Beautiful Rich CLI with fallback to plain output

Code By : YSNRFD
Telegram: @ysnrfd | @ysnrfd3
Bale: @ysnrfd | @ysnrfd2
GitHub  : ysnrfd
HuggingFace: ysn-rfd
--------------------
UNDER YSNRFD LICENSE
--------------------
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text

    HAS_RICH = True
    _RichConsole = Console
except ImportError:
    HAS_RICH = False
    _RichConsole = None


try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        LCMScheduler,
        UniPCMultistepScheduler,
        KDPM2DiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        HeunDiscreteScheduler,
    )
except ImportError as exc:
    print(f"ERROR: Failed to import diffusers: {exc}")
    print("Install with:  pip install --upgrade diffusers transformers accelerate safetensors torch torchvision")
    sys.exit(1)

import diffusers


VERSION = "3.0.0"
LICENSE_TEXT = "UNDER YSNRFD LICENSE"
DEFAULT_OUTPUT_DIR = Path("output")
CONFIG_DIR = Path("configs")
HISTORY_FILE = Path("prompt_history.json")

SCHEDULER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ddim":       {"class": DDIMScheduler,                       "label": "DDIM",             "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "pndm":       {"class": PNDMScheduler,                       "label": "PNDM",             "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "lms":        {"class": LMSDiscreteScheduler,                "label": "LMS Discrete",     "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "euler":      {"class": EulerDiscreteScheduler,              "label": "Euler",            "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "euler-a":    {"class": EulerAncestralDiscreteScheduler,     "label": "Euler Ancestral",  "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "dpm++2m":    {"class": DPMSolverMultistepScheduler,         "label": "DPM++ 2M",         "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "dpm++sde":   {"class": DPMSolverSDEScheduler,               "label": "DPM++ SDE",        "default_cfg": 7.0,  "min_steps": 10, "lcm_ok": False},
    "dpm++2m-sde":{"class": DPMSolverMultistepScheduler,         "label": "DPM++ 2M SDE",     "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "unipc":      {"class": UniPCMultistepScheduler,             "label": "UniPC",            "default_cfg": 7.5,  "min_steps": 5,  "lcm_ok": False},
    "heun":       {"class": HeunDiscreteScheduler,               "label": "Heun",             "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "kdpm2":      {"class": KDPM2DiscreteScheduler,              "label": "KDPM2",            "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "kdpm2-a":    {"class": KDPM2AncestralDiscreteScheduler,     "label": "KDPM2 Ancestral",  "default_cfg": 7.5,  "min_steps": 10, "lcm_ok": False},
    "lcm":        {"class": LCMScheduler,                         "label": "LCM",              "default_cfg": 1.5,  "min_steps": 4,  "lcm_ok": True},
}

# default recommended steps per scheduler
SCHEDULER_DEFAULT_STEPS = {
    "ddim": 20, "pndm": 20, "lms": 20, "euler": 20,
    "euler-a": 20, "dpm++2m": 20, "dpm++sde": 20, "dpm++2m-sde": 20,
    "unipc": 20, "heun": 20, "kdpm2": 20, "kdpm2-a": 20, "lcm": 8,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "photorealistic": {
        "prompt_suffix": ", photorealistic, 8k uhd, high quality, professional photography, dslr",
        "negative_prompt": "cartoon, drawing, illustration, anime, 3d render, blurry, low quality, watermark, text",
        "cfg_scale": 7.5,
        "steps": 25,
        "scheduler": "euler-a",
    },
    "anime": {
        "prompt_suffix": ", anime style, vibrant colors, cel-shaded, detailed anime art",
        "negative_prompt": "photorealistic, 3d, realistic, photograph, blurry, low quality",
        "cfg_scale": 7.0,
        "steps": 20,
        "scheduler": "euler",
    },
    "digital-art": {
        "prompt_suffix": ", digital art, concept art, trending on artstation, highly detailed",
        "negative_prompt": "photograph, realistic, 3d render, blurry, low quality, watermark",
        "cfg_scale": 8.0,
        "steps": 30,
        "scheduler": "dpm++2m",
    },
    "3d-render": {
        "prompt_suffix": ", 3d render, blender, octane render, unreal engine 5, volumetric lighting",
        "negative_prompt": "photograph, drawing, anime, blurry, low quality, flat shading",
        "cfg_scale": 9.0,
        "steps": 30,
        "scheduler": "dpm++2m",
    },
    "painting": {
        "prompt_suffix": ", oil painting, masterpiece, classical art, rich colors, impasto",
        "negative_prompt": "photograph, 3d, digital art, anime, blurry, low quality",
        "cfg_scale": 8.0,
        "steps": 25,
        "scheduler": "euler-a",
    },
}

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "3:2":  (1216, 832),
    "2:3":  (832, 1216),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}

ASPECT_RATIOS_SD15: Dict[str, Tuple[int, int]] = {
    "1:1":  (512, 512),
    "3:2":  (640, 416),
    "2:3":  (416, 640),
    "4:3":  (576, 448),
    "3:4":  (448, 576),
    "16:9": (704, 384),
    "9:16": (384, 704),
}



@dataclass
class GenerationConfig:
    """Holds all parameters for a generation run."""
    model_path: str = ""
    model_type: str = "auto"          # "sd15", "sdxl", "auto"
    is_img2img: bool = False
    refiner_path: str = ""
    refiner_strength: float = 0.3     # high_noise_frac for SDXL refiner

    # LoRA
    lora_paths: List[str] = field(default_factory=list)
    lora_weights: List[float] = field(default_factory=list)

    # Scheduler and sampling
    scheduler: str = "lcm"
    steps: int = 0                    # 0 = use scheduler default
    cfg_scale: float = 0.0            # 0 = use scheduler default
    seed: int = -1                    # -1 = random

    # Image dimensions
    width: int = 0                    # 0 = use model default
    height: int = 0

    # Prompts
    prompt: str = "A beautiful landscape with mountains and a lake, ultra-detailed, realistic, 4k"
    negative_prompt: str = ""

    # img2img
    image_path: str = ""
    strength: float = 0.75

    # Batch
    batch_size: int = 1

    # Flags
    skip_quantization: bool = False
    no_warmup: bool = False
    post_process: bool = False
    save_report: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    # Computed / runtime
    _resolved_model_type: str = field(default="", repr=False)
    _actual_seed: int = field(default=0, repr=False)

    def resolve_defaults(self) -> None:
        """Fill in zero/empty fields with sensible defaults."""
        # Resolve model type
        if self.model_type == "auto":
            self._resolved_model_type = detect_model_type(self.model_path)
        else:
            self._resolved_model_type = self.model_type

        is_sdxl = self._resolved_model_type == "sdxl"

        # Default dimensions
        if self.width == 0 or self.height == 0:
            if is_sdxl:
                self.width, self.height = 1024, 1024
            else:
                self.width, self.height = 512, 512

        # Ensure dimensions are multiples of 8
        self.width = max(64, (self.width // 8) * 8)
        self.height = max(64, (self.height // 8) * 8)

        # Default steps
        if self.steps <= 0:
            self.steps = SCHEDULER_DEFAULT_STEPS.get(self.scheduler, 20)

        # Default CFG
        if self.cfg_scale <= 0.0:
            sched_info = SCHEDULER_REGISTRY.get(self.scheduler)
            self.cfg_scale = sched_info["default_cfg"] if sched_info else 7.5

        # Seed
        if self.seed < 0:
            self._actual_seed = random.randint(0, 2**32 - 1)
        else:
            self._actual_seed = self.seed

        # LoRA weights default
        while len(self.lora_weights) < len(self.lora_paths):
            self.lora_weights.append(1.0)

        # Output dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)



class Console:
    """Unified console that uses Rich when available, plain text otherwise."""

    def __init__(self):
        if HAS_RICH:
            self._console = _RichConsole()
            self._use_rich = True
        else:
            self._console = None
            self._use_rich = False

    def banner(self, text: str, style: str = "bold cyan") -> None:
        if self._use_rich:
            self._console.print(Panel(text, style=style, expand=False))
        else:
            print(f"\n{'='*60}")
            print(text)
            print(f"{'='*60}")

    def section(self, title: str) -> None:
        if self._use_rich:
            self._console.rule(f"[bold]{title}")
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")

    def info(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(f"[cyan]info[/]  {msg}")
        else:
            print(f"  INFO: {msg}")

    def success(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(f"[green]okay[/]  {msg}")
        else:
            print(f"  OK: {msg}")

    def warning(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(f"[yellow]warning[/]  {msg}")
        else:
            print(f"  WARNING: {msg}")

    def error(self, msg: str) -> None:
        if self._use_rich:
            self._console.print(f"[red]error[/]  {msg}")
        else:
            print(f"  ERROR: {msg}")

    def print(self, msg: str = "") -> None:
        if self._use_rich:
            self._console.print(msg)
        else:
            print(msg)

    def table(self, headers: List[str], rows: List[List[str]], title: str = "") -> None:
        if self._use_rich:
            t = Table(title=title, show_lines=True)
            for h in headers:
                t.add_column(h, justify="left")
            for row in rows:
                t.add_row(*row)
            self._console.print(t)
        else:
            if title:
                print(f"\n  {title}")
            col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(f"  {header_line}")
            print(f"  {'-' * len(header_line)}")
            for row in rows:
                print(f"  {' | '.join(c.ljust(w) for c, w in zip(row, col_widths))}")

    def prompt(self, label: str, default: str = "") -> str:
        if self._use_rich:
            return Prompt.ask(f"[bold]{label}", default=default)
        val = input(f"{label} [{default}]: ").strip()
        return val if val else default

    def confirm(self, label: str, default: bool = True) -> bool:
        if self._use_rich:
            return Confirm.ask(f"[bold]{label}", default=default)
        hint = "Y/n" if default else "y/N"
        val = input(f"{label} ({hint}): ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes")

    def choice(self, label: str, options: List[str], default: str = "") -> str:
        for i, opt in enumerate(options, 1):
            self.print(f"  {i}. {opt}")
        return self.prompt(label, default=default)


con = Console()



def detect_model_type(model_path: str) -> str:
    """Auto-detect whether a model is SD 1.5 or SDXL based on filename/config."""
    name = model_path.lower()
    if "sdxl" in name or "xl" in name:
        return "sdxl"

    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    idx = json.load(f)
                # SDXL uses StableDiffusionXLPipeline class
                class_name = idx.get("_class_name", "")
                if "XL" in class_name:
                    return "sdxl"
                # SDXL UNet has 2810MB vs SD 1.5 1700MB | check architecture
                unet_config = os.path.join(model_path, "unet", "config.json")
                if os.path.exists(unet_config):
                    with open(unet_config, "r") as f:
                        ucfg = json.load(f)
                    # SDXL has addition_embed_type = "text_time"
                    if ucfg.get("addition_embed_type") == "text_time":
                        return "sdxl"
                    # SDXL has dual cross attention
                    if ucfg.get("dual_cross_attention", False):
                        return "sdxl"
                    # SDXL block out channels typically [2,4,4,4] vs [1,2,4,4]
                    boc = ucfg.get("block_out_channels", [])
                    if boc and boc[0] == 2:
                        return "sdxl"
            except Exception:
                pass

    # For safetensors files, try loading minimal config
    if name.endswith(".safetensors"):
        try:
            from safetensors.torch import safe_open
            with safe_open(model_path, framework="pt") as f:
                keys = f.keys()
                # SDXL models contain specific keys
                for k in keys:
                    if "add_embedding" in k or "down_blocks.2.downsamplers.0.op.weight" in k:
                        return "sdxl"
                # Heuristic: count parameters — SDXL UNet is 2.6B params
                unet_keys = [k for k in keys if k.startswith("model.diffusion_model.") or
                             ("down_blocks" in k or "up_blocks" in k or "mid_block" in k)]
                if len(unet_keys) > 1500:
                    return "sdxl"
        except Exception:
            pass

    return "sd15"


def get_memory_usage() -> float:
    """Return current process RSS in GB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    return 0.0


def get_system_memory() -> Tuple[float, float, float]:
    """Return (available_gb, total_gb, percent) of system memory."""
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        return vm.available / (1024 ** 3), vm.total / (1024 ** 3), vm.percent
    return 0.0, 0.0, 0.0


def ensure_multiple_of_8(val: int, minimum: int = 64, maximum: int = 4096) -> int:
    """Clamp a dimension to a multiple of 8 within [minimum, maximum]."""
    val = (val // 8) * 8
    return max(minimum, min(maximum, val))


def safe_save_image(image: Image.Image, path: str, quality: int = 95) -> str:
    """Save image, creating directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality)
    return path



def load_pipeline(config: GenerationConfig) -> Any:
    """Load the appropriate pipeline based on config."""
    con.section("Loading Model")
    is_sdxl = config._resolved_model_type == "sdxl"
    model_label = "SDXL" if is_sdxl else "SD 1.5"
    con.info(f"Model type: {model_label}")
    con.info(f"Path: {config.model_path}")

    if not os.path.exists(config.model_path):
        con.error(f"Model file not found: {config.model_path}")
        raise FileNotFoundError(config.model_path)

    common_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float32,
        "use_safetensors": True if config.model_path.endswith(".safetensors") else None,
    }
    # Remove None values
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

    # For local safetensors files, use from_single_file
    use_single_file = config.model_path.endswith(".safetensors") or config.model_path.endswith(".ckpt")

    try:
        if config.is_img2img:
            if is_sdxl:
                PipeClass = StableDiffusionXLImg2ImgPipeline
            else:
                PipeClass = StableDiffusionImg2ImgPipeline
            con.info("Mode: Image-to-Image")
        else:
            if is_sdxl:
                PipeClass = StableDiffusionXLPipeline
            else:
                PipeClass = StableDiffusionPipeline
            con.info("Mode: Text-to-Image")

        if use_single_file:
            con.info("Loading from single file...")
            pipe = PipeClass.from_single_file(
                config.model_path,
                **common_kwargs,
            )
        else:
            con.info("Loading from HuggingFace / local directory...")
            pipe = PipeClass.from_pretrained(
                config.model_path,
                **common_kwargs,
            )

        # Disable safety checker if present
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False

    except Exception as exc:
        con.error(f"Failed to load model: {exc}")
        con.info("Trying alternative loading method...")
        try:
            if use_single_file:
                pipe = PipeClass.from_single_file(
                    config.model_path,
                    torch_dtype=torch.float32,
                )
            else:
                pipe = PipeClass.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float32,
                )
        except Exception as exc2:
            con.error(f"Alternative loading also failed: {exc2}")
            raise

    # Configure scheduler
    sched_info = SCHEDULER_REGISTRY.get(config.scheduler)
    if sched_info is None:
        con.warning(f"Unknown scheduler '{config.scheduler}', falling back to 'euler-a'")
        sched_info = SCHEDULER_REGISTRY["euler-a"]
        config.scheduler = "euler-a"

    con.info(f"Scheduler: {sched_info['label']}")
    try:
        pipe.scheduler = sched_info["class"].from_config(pipe.scheduler.config)
    except Exception as exc:
        con.warning(f"Scheduler config failed ({exc}), using default config")
        pipe.scheduler = sched_info["class"]()

    # Move to CPU
    pipe = pipe.to("cpu")
    con.success(f"Model loaded: {model_label} with {sched_info['label']} scheduler")

    return pipe


def load_refiner_pipeline(refiner_path: str) -> Any:
    """Load the SDXL refiner model."""
    if not refiner_path:
        return None
    con.section("Loading SDXL Refiner")
    con.info(f"Refiner path: {refiner_path}")

    if not os.path.exists(refiner_path):
        con.error(f"Refiner model not found: {refiner_path}")
        return None

    use_single_file = refiner_path.endswith(".safetensors") or refiner_path.endswith(".ckpt")

    try:
        if use_single_file:
            refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                refiner_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        else:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_path,
                torch_dtype=torch.float32,
            )

        if hasattr(refiner, "safety_checker") and refiner.safety_checker is not None:
            refiner.safety_checker = None
        if hasattr(refiner, "requires_safety_checker"):
            refiner.requires_safety_checker = False

        refiner = refiner.to("cpu")
        con.success("SDXL Refiner loaded")
        return refiner
    except Exception as exc:
        con.error(f"Failed to load refiner: {exc}")
        return None



def apply_cpu_optimizations(pipe: Any, config: GenerationConfig) -> Any:
    """Apply CPU-specific memory and speed optimizations."""
    con.section("Applying CPU Optimizations")

    has_lora = bool(config.lora_paths)
    is_img2img = config.is_img2img

    # Quantization strategy
    if is_img2img:
        quantize_vae = False
        quantize_unet = False
        quantize_text_encoder = False
        con.info("img2img mode: all quantization DISABLED to preserve quality")
    elif has_lora:
        quantize_vae = True
        quantize_unet = False
        quantize_text_encoder = False
        con.info("LoRA active: quantizing VAE only (UNet skipped for LoRA compatibility)")
    else:
        quantize_vae = True
        quantize_unet = True
        quantize_text_encoder = True
        con.info("Full 8-bit quantization enabled")

    # Apply quantization
    if quantize_vae and hasattr(pipe, "vae"):
        try:
            pipe.vae = torch.quantization.quantize_dynamic(
                pipe.vae, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            con.success("VAE quantized to 8-bit (~150 MB saved)")
        except Exception as exc:
            con.warning(f"VAE quantization failed: {exc}")

    if quantize_unet and hasattr(pipe, "unet"):
        try:
            pipe.unet = torch.quantization.quantize_dynamic(
                pipe.unet, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            con.success("UNet quantized to 8-bit")
        except Exception as exc:
            con.warning(f"UNet quantization failed: {exc}")

    if quantize_text_encoder and hasattr(pipe, "text_encoder"):
        try:
            pipe.text_encoder = torch.quantization.quantize_dynamic(
                pipe.text_encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
            con.success("Text encoder quantized to 8-bit")
        except Exception as exc:
            con.warning(f"Text encoder quantization failed: {exc}")

    # SDXL has a second text encoder
    if quantize_text_encoder and hasattr(pipe, "text_encoder_2"):
        try:
            pipe.text_encoder_2 = torch.quantization.quantize_dynamic(
                pipe.text_encoder_2, {torch.nn.Linear}, dtype=torch.qint8
            )
            con.success("Text encoder 2 quantized to 8-bit")
        except Exception as exc:
            con.warning(f"Text encoder 2 quantization failed: {exc}")

    # Attention slicing — significant memory savings on CPU
    try:
        pipe.enable_attention_slicing("max")
        con.success("Attention slicing enabled (20-30% memory reduction)")
    except Exception as exc:
        con.warning(f"Attention slicing failed: {exc}")

    # VAE slicing
    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            con.success("VAE slicing enabled (15-25% memory reduction)")
    except Exception as exc:
        con.warning(f"VAE slicing failed: {exc}")

    # CPU threading
    cpu_cores = max(1, os.cpu_count() or 1)
    try:
        torch.set_num_threads(cpu_cores)
        torch.set_num_interop_threads(max(1, min(4, cpu_cores)))
        con.info(f"CPU threads: {cpu_cores}")
    except Exception:
        pass

    # Disable CUDA backends
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    # Enable VAE tiling for very large images (SDXL default 1024+)
    if config.width >= 1024 or config.height >= 1024:
        try:
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                con.success("VAE tiling enabled for high-resolution generation")
        except Exception:
            pass

    # Report memory
    mem = get_memory_usage()
    if mem > 0:
        con.info(f"Process memory after optimization: {mem:.2f} GB")

    return pipe



def load_loras(pipe: Any, config: GenerationConfig) -> bool:
    """Load and activate multiple LoRAs."""
    if not config.lora_paths:
        return False

    con.section("Loading LoRAs")
    adapter_names: List[str] = []
    valid_weights: List[float] = []

    for i, (lora_path, weight) in enumerate(zip(config.lora_paths, config.lora_weights)):
        adapter_name = f"lora_{i+1}"
        con.info(f"LoRA #{i+1}: {os.path.basename(lora_path)} (strength: {weight:.2f})")

        if not os.path.exists(lora_path):
            con.error(f"File not found: {lora_path}")
            continue

        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            valid_weights.append(weight)
            con.success(f"LoRA #{i+1} loaded as '{adapter_name}'")
        except Exception as exc:
            con.error(f"Failed to load LoRA #{i+1}: {exc}")
            continue

    if not adapter_names:
        con.error("No LoRAs loaded successfully")
        return False

    # Activate all loaded LoRAs
    try:
        pipe.set_adapters(adapter_names, valid_weights)
        con.success(f"{len(adapter_names)} LoRA(s) activated: {adapter_names} with weights {valid_weights}")
        return True
    except Exception as exc:
        con.error(f"Failed to activate LoRAs: {exc}")
        return False


def unload_loras(pipe: Any) -> None:
    """Unload all LoRA adapters."""
    try:
        adapters = pipe.get_list_adapters()
        for name in adapters:
            try:
                pipe.unload_lora_weights(name)
            except Exception:
                pass
        con.info("All LoRAs unloaded")
    except Exception:
        pass



def warm_up(pipe: Any, config: GenerationConfig) -> None:
    """Quick warm-up pass to initialise lazy-loaded components."""
    if config.no_warmup:
        return

    con.section("Warming Up Model")
    start = time.time()
    try:
        if config.is_img2img:
            import numpy as np
            warmup_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
            pipe(
                prompt="warmup",
                image=warmup_img,
                strength=0.5,
                num_inference_steps=1,
                guidance_scale=1.0,
                output_type="np",
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
        else:
            pipe(
                prompt="warmup",
                width=128,
                height=128,
                num_inference_steps=1,
                guidance_scale=1.0,
                output_type="np",
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
        elapsed = time.time() - start
        con.success(f"Warm-up completed in {elapsed:.1f}s")
    except Exception as exc:
        con.warning(f"Warm-up skipped ({exc})")


def generate_text2img(pipe: Any, config: GenerationConfig, refiner: Any = None) -> List[Tuple[Image.Image, str]]:
    """Text-to-image generation. Returns list of (image, output_path)."""
    con.section("Generating Images (Text-to-Image)")

    results: List[Tuple[Image.Image, str]] = []
    is_sdxl = config._resolved_model_type == "sdxl"

    for batch_idx in range(config.batch_size):
        seed = config._actual_seed + batch_idx
        con.info(f"Image {batch_idx + 1}/{config.batch_size}  |  seed={seed}")

        gc.collect()

        gen_kwargs: Dict[str, Any] = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt or None,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.steps,
            "guidance_scale": config.cfg_scale,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "output_type": "pil",
        }

        # SDXL-specific: denoising end for refiner
        if is_sdxl and refiner is not None:
            gen_kwargs["denoising_end"] = config.refiner_strength
            gen_kwargs["output_type"] = "pil"

        # Callback for progress
        if HAS_RICH:
            from rich.progress import Progress as RProgress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            with RProgress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=con._console,
            ) as progress:
                task = progress.add_task("Generating", total=config.steps)

                def step_callback(pipes, step, timestep, callback_kwargs):
                    progress.update(task, completed=step + 1)
                    return callback_kwargs

                gen_kwargs["callback_on_step_end"] = step_callback
                output = pipe(**gen_kwargs)
        else:
            output = pipe(**gen_kwargs)

        # Handle SDXL refiner
        if is_sdxl and refiner is not None:
            con.info("Running SDXL refiner...")
            image = output.images[0]
            refiner_kwargs = {
                "prompt": config.prompt,
                "negative_prompt": config.negative_prompt or None,
                "image": image,
                "strength": config.refiner_strength,
                "num_inference_steps": config.steps,
                "guidance_scale": config.cfg_scale,
                "generator": torch.Generator(device="cpu").manual_seed(seed),
                "output_type": "pil",
            }
            refiner_output = refiner(**refiner_kwargs)
            image = refiner_output.images[0]
        else:
            image = output.images[0]

        # Post-processing
        if config.post_process:
            image = apply_post_processing(image)

        # Save
        ts = int(time.time())
        suffix = f"_sdxl" if is_sdxl else ""
        lora_suffix = "_lora" if config.lora_paths else ""
        out_name = f"txt2img{suffix}{lora_suffix}_{config.width}x{config.height}_{ts}_seed{seed}.png"
        out_path = os.path.join(config.output_dir, out_name)
        safe_save_image(image, out_path)

        con.success(f"Saved: {out_path}")
        results.append((image, out_path))

    return results


def generate_img2img(pipe: Any, config: GenerationConfig) -> List[Tuple[Image.Image, str]]:
    """Image-to-image generation. Returns list of (image, output_path)."""
    con.section("Generating Images (Image-to-Image)")

    if not config.image_path or not os.path.exists(config.image_path):
        con.error(f"Input image not found: {config.image_path}")
        return []

    # Load and preprocess input image
    init_image = load_image(config.image_path, target_size=(config.width, config.height))
    if init_image is None:
        return []

    is_sdxl = config._resolved_model_type == "sdxl"
    results: List[Tuple[Image.Image, str]] = []

    for batch_idx in range(config.batch_size):
        seed = config._actual_seed + batch_idx
        con.info(f"Image {batch_idx + 1}/{config.batch_size}  |  seed={seed}  |  strength={config.strength:.2f}")

        gc.collect()

        gen_kwargs: Dict[str, Any] = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt or None,
            "image": init_image,
            "strength": config.strength,
            "num_inference_steps": config.steps,
            "guidance_scale": config.cfg_scale,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "output_type": "pil",
        }

        # Callback for progress
        if HAS_RICH:
            from rich.progress import Progress as RProgress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
            with RProgress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=con._console,
            ) as progress:
                task = progress.add_task("Generating", total=config.steps)

                def step_callback(pipes, step, timestep, callback_kwargs):
                    progress.update(task, completed=step + 1)
                    return callback_kwargs

                gen_kwargs["callback_on_step_end"] = step_callback
                output = pipe(**gen_kwargs)
        else:
            output = pipe(**gen_kwargs)

        image = output.images[0]

        # Post-processing
        if config.post_process:
            image = apply_post_processing(image)

        # Save
        ts = int(time.time())
        suffix = f"_sdxl" if is_sdxl else ""
        lora_suffix = "_lora" if config.lora_paths else ""
        out_name = f"img2img{suffix}{lora_suffix}_{ts}_seed{seed}_str{config.strength:.2f}.png"
        out_path = os.path.join(config.output_dir, out_name)
        safe_save_image(image, out_path)

        con.success(f"Saved: {out_path}")
        results.append((image, out_path))

    return results



def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """Load and preprocess an image for img2img."""
    try:
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        if target_size is not None:
            tw, th = target_size
            ratio = min(tw / orig_w, th / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            # Center crop to exact target size
            left = (new_w - tw) // 2
            top = (new_h - th) // 2
            image = image.crop((left, top, left + tw, top + th))
            con.info(f"Image: {orig_w}x{orig_h} -> {tw}x{th}")
        else:
            # Ensure dimensions are multiples of 8
            new_w = max(64, (orig_w // 8) * 8)
            new_h = max(64, (orig_h // 8) * 8)
            left = (orig_w - new_w) // 2
            top = (orig_h - new_h) // 2
            image = image.crop((left, top, left + new_w, top + new_h))
            con.info(f"Image: {orig_w}x{orig_h} -> {new_w}x{new_h}")

        return image
    except Exception as exc:
        con.error(f"Failed to load image: {exc}")
        return None


def apply_post_processing(image: Image.Image) -> Image.Image:
    """Apply subtle post-processing enhancements."""
    con.info("Applying post-processing...")
    try:
        image = ImageEnhance.Contrast(image).enhance(1.1)
        image = ImageEnhance.Sharpness(image).enhance(1.2)
        image = ImageEnhance.Color(image).enhance(1.05)
        image = image.filter(ImageFilter.SMOOTH)
        con.success("Post-processing applied")
    except Exception as exc:
        con.warning(f"Post-processing failed: {exc}")
    return image



def save_config(config: GenerationConfig, filename: str = "") -> bool:
    """Save generation config to a JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = f"config_{int(time.time())}.json"
    if not filename.endswith(".json"):
        filename += ".json"
    path = CONFIG_DIR / filename
    try:
        data = asdict(config)
        # Remove private fields
        data.pop("_resolved_model_type", None)
        data.pop("_actual_seed", None)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        con.success(f"Config saved: {path}")
        return True
    except Exception as exc:
        con.error(f"Failed to save config: {exc}")
        return False


def load_config(filename: str) -> Optional[GenerationConfig]:
    """Load generation config from a JSON file."""
    path = Path(filename)
    if not path.exists():
        path = CONFIG_DIR / filename
    if not path.exists():
        con.error(f"Config file not found: {filename}")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Remove fields that dont belong in GenerationConfig
        data.pop("_resolved_model_type", None)
        data.pop("_actual_seed", None)
        config = GenerationConfig(**{k: v for k, v in data.items() if k in GenerationConfig.__dataclass_fields__})
        con.success(f"Config loaded: {path}")
        return config
    except Exception as exc:
        con.error(f"Failed to load config: {exc}")
        return None


def list_configs() -> List[str]:
    """List available configuration files."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(str(p.name) for p in CONFIG_DIR.glob("*.json"))




def load_prompt_history() -> List[Dict[str, Any]]:
    """Load prompt history from file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_prompt_to_history(prompt: str, negative_prompt: str = "") -> None:
    """Save a prompt to history (max 100 entries)."""
    history = load_prompt_history()
    history.insert(0, {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": int(time.time()),
    })
    history = history[:100]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass



def generate_report(config: GenerationConfig, output_path: str, elapsed: float) -> str:
    """Generate a detailed text report of the generation."""
    report_path = output_path.replace(".png", "_report.txt")
    is_sdxl = config._resolved_model_type == "sdxl"

    lines = [
        "=" * 60,
        "STABLE DIFFUSION GENERATION REPORT",
        "=" * 60,
        f"Timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model        : {os.path.basename(config.model_path)}",
        f"Model Type   : {'SDXL' if is_sdxl else 'SD 1.5'}",
        f"Mode         : {'Image-to-Image' if config.is_img2img else 'Text-to-Image'}",
        "",
        "--- Generation Parameters ---",
        f"Prompt         : {config.prompt}",
        f"Negative Prompt: {config.negative_prompt or '(none)'}",
        f"Scheduler      : {SCHEDULER_REGISTRY.get(config.scheduler, {}).get('label', config.scheduler)}",
        f"CFG Scale      : {config.cfg_scale:.1f}",
        f"Steps          : {config.steps}",
        f"Seed           : {config._actual_seed}",
        f"Image Size     : {config.width}x{config.height}",
        f"Batch Size     : {config.batch_size}",
    ]

    if config.is_img2img:
        lines.append(f"Image Strength : {config.strength:.2f}")
        lines.append(f"Input Image    : {config.image_path}")

    if config.lora_paths:
        lines.append("")
        lines.append("--- LoRA Configuration ---")
        for i, (p, w) in enumerate(zip(config.lora_paths, config.lora_weights)):
            lines.append(f"  LoRA #{i+1}: {os.path.basename(p)} (weight: {w:.2f})")

    if config.refiner_path:
        lines.append(f"Refiner        : {os.path.basename(config.refiner_path)}")
        lines.append(f"Refiner Str.   : {config.refiner_strength:.2f}")

    mem = get_memory_usage()
    lines += [
        "",
        "--- Performance ---",
        f"Generation Time : {elapsed:.2f}s",
        f"Memory Usage    : {mem:.2f} GB",
        f"Output Path     : {output_path}",
        "",
        "--- System ---",
        f"Python   : {sys.version.split()[0]}",
        f"Diffusers: {diffusers.__version__}",
        f"Torch    : {torch.__version__}",
        f"CPU Cores: {os.cpu_count()}",
    ]

    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        lines.append(f"System RAM: {vm.total / (1024**3):.1f} GB total, {vm.available / (1024**3):.1f} GB available")

    lines.append("=" * 60)

    try:
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        con.success(f"Report saved: {report_path}")
    except Exception as exc:
        con.warning(f"Failed to save report: {exc}")

    return report_path


def check_system_resources(required_gb: float = 2.0) -> bool:
    """Check if the system has enough resources. Returns True if OK."""
    if not HAS_PSUTIL:
        con.warning("psutil not installed — skipping resource check")
        return True

    vm = psutil.virtual_memory()
    available = vm.available / (1024 ** 3)
    cores = os.cpu_count() or 1

    con.info(f"Available RAM: {available:.2f} GB  |  CPU cores: {cores}")

    if available < required_gb:
        con.warning(f"Low memory ({available:.2f} GB available, {required_gb:.1f} GB recommended)")
        return False
    if cores < 2:
        con.warning("Very few CPU cores — generation will be extremely slow")

    return True



def cleanup_pipeline(pipe: Any) -> None:
    """Thoroughly unload a pipeline from memory."""
    con.info("Cleaning up pipeline...")
    try:
        # Unload LoRAs first
        unload_loras(pipe)

        # Move components to CPU then delete
        components = ["unet", "text_encoder", "text_encoder_2", "vae", "scheduler",
                      "tokenizer", "tokenizer_2", "feature_extractor", "image_encoder"]
        for comp in components:
            if hasattr(pipe, comp):
                try:
                    delattr(pipe, comp)
                except Exception:
                    pass

        del pipe
        gc.collect()
        mem = get_memory_usage()
        if mem > 0:
            con.info(f"Memory after cleanup: {mem:.2f} GB")
        con.success("Pipeline cleaned up")
    except Exception as exc:
        con.warning(f"Cleanup error: {exc}")



def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser."""
    parser = argparse.ArgumentParser(
        prog="sd-cpu-pipeline",
        description="CPU-Optimized Stable Diffusion Pipeline — SD 1.5 & SDXL",
        epilog=f"Version {VERSION} | {LICENSE_TEXT}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── generate ──
    gen = sub.add_parser("generate", help="Generate images", formatter_class=argparse.RawDescriptionHelpFormatter)
    gen.add_argument("model", help="Path to model file (.safetensors, .ckpt) or HuggingFace repo ID")

    # Model type
    gen.add_argument("--model-type", choices=["sd15", "sdxl", "auto"], default="auto",
                     help="Force model type (default: auto-detect)")

    # Mode
    gen.add_argument("--img2img", action="store_true", help="Image-to-image mode")
    gen.add_argument("--input-image", type=str, default="", help="Input image for img2img")
    gen.add_argument("--strength", type=float, default=0.75, help="img2img strength (0.1-1.0)")

    # SDXL refiner
    gen.add_argument("--refiner", type=str, default="", help="Path to SDXL refiner model")
    gen.add_argument("--refiner-strength", type=float, default=0.3, help="Refiner high_noise_frac (0.1-1.0)")

    # LoRA
    gen.add_argument("--lora", nargs="*", default=[], help="Paths to LoRA weights")
    gen.add_argument("--lora-weight", nargs="*", type=float, default=[], help="LoRA strengths (0.0-2.0)")

    # Scheduler
    sched_names = list(SCHEDULER_REGISTRY.keys())
    gen.add_argument("--scheduler", choices=sched_names, default="euler-a",
                     help="Sampling scheduler (default: euler-a)")
    gen.add_argument("--steps", type=int, default=0, help="Inference steps (0=scheduler default)")
    gen.add_argument("--cfg", type=float, default=0.0, help="CFG guidance scale (0=scheduler default)")

    # Seed & batch
    gen.add_argument("--seed", type=int, default=-1, help="Random seed (-1=random)")
    gen.add_argument("--batch", type=int, default=1, help="Number of images to generate")

    # Dimensions
    gen.add_argument("--width", type=int, default=0, help="Image width (0=model default)")
    gen.add_argument("--height", type=int, default=0, help="Image height (0=model default)")
    gen.add_argument("--aspect", type=str, default="", help="Aspect ratio (e.g. 16:9, 1:1, 3:2)")

    # Prompts
    gen.add_argument("--prompt", type=str, default="", help="Text prompt")
    gen.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")

    # Presets
    gen.add_argument("--preset", choices=list(PRESETS.keys()), default="", help="Apply a style preset")

    # Config
    gen.add_argument("--config", type=str, default="", help="Load generation config from JSON file")
    gen.add_argument("--save-config", type=str, default="", help="Save generation config to JSON file")

    # Flags
    gen.add_argument("--no-warmup", action="store_true", help="Skip model warm-up")
    gen.add_argument("--post-process", action="store_true", help="Apply post-processing")
    gen.add_argument("--report", action="store_true", help="Generate detailed text report")
    gen.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    # ── info ──
    info = sub.add_parser("info", help="Show system info and model details")
    info.add_argument("model", nargs="?", default="", help="Path to model for type detection")

    # ── interactive ──
    inter = sub.add_parser("interactive", help="Launch interactive menu mode")
    inter.add_argument("--model", type=str, default="", help="Pre-load model path")

    # ── schedulers ──
    sub.add_parser("schedulers", help="List available schedulers")

    # ── presets ──
    sub.add_parser("presets", help="List available presets")

    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle the 'generate' subcommand."""
    # Build config from args
    config = GenerationConfig(
        model_path=args.model,
        model_type=args.model_type,
        is_img2img=args.img2img,
        refiner_path=getattr(args, "refiner", ""),
        refiner_strength=getattr(args, "refiner_strength", 0.3),
        lora_paths=args.lora,
        lora_weights=args.lora_weight,
        scheduler=args.scheduler,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=getattr(args, "input_image", ""),
        strength=getattr(args, "strength", 0.75),
        batch_size=max(1, getattr(args, "batch", 1)),
        no_warmup=getattr(args, "no_warmup", False),
        post_process=getattr(args, "post_process", False),
        save_report=getattr(args, "report", False),
        output_dir=getattr(args, "output_dir", str(DEFAULT_OUTPUT_DIR)),
    )

    # Load from config file if specified
    if args.config:
        loaded = load_config(args.config)
        if loaded:
            # Override with CLI args where provided
            for k, v in asdict(config).items():
                if v and k not in ("_resolved_model_type", "_actual_seed"):
                    try:
                        setattr(loaded, k, v)
                    except Exception:
                        pass
            config = loaded

    # Apply preset
    if args.preset:
        preset = PRESETS[args.preset]
        if not config.prompt:
            config.prompt = "masterpiece, best quality"
        config.prompt += preset["prompt_suffix"]
        if not config.negative_prompt:
            config.negative_prompt = preset["negative_prompt"]
        if config.cfg_scale <= 0:
            config.cfg_scale = preset["cfg_scale"]
        if config.steps <= 0:
            config.steps = preset["steps"]
        if config.scheduler == "euler-a" and args.scheduler == "euler-a":
            config.scheduler = preset.get("scheduler", "euler-a")

    # Apply aspect ratio
    if args.aspect:
        is_sdxl = config.model_type == "sdxl" or (config.model_type == "auto" and detect_model_type(config.model_path) == "sdxl")
        ratios = ASPECT_RATIOS if is_sdxl else ASPECT_RATIOS_SD15
        if args.aspect in ratios:
            config.width, config.height = ratios[args.aspect]
        else:
            con.warning(f"Unknown aspect ratio '{args.aspect}', ignoring")

    # Validate img2img
    if config.is_img2img and not config.image_path:
        con.error("img2img mode requires --input-image")
        return 1

    # Prompt from history if not provided
    if not config.prompt:
        config.prompt = "A beautiful landscape with mountains and a lake, ultra-detailed, realistic, 4k"

    # Resolve all defaults
    config.resolve_defaults()

    # Print summary
    _print_config_summary(config)

    # Check resources
    required_gb = 2.0
    if config._resolved_model_type == "sdxl":
        required_gb = 4.0
    if config.lora_paths:
        required_gb += 0.5
    check_system_resources(required_gb)

    # Save config if requested
    if args.save_config:
        save_config(config, args.save_config)

    # Load model
    pipe = load_pipeline(config)

    # Load refiner if specified
    refiner = None
    if config.refiner_path and config._resolved_model_type == "sdxl":
        refiner = load_refiner_pipeline(config.refiner_path)

    # Warm up
    warm_up(pipe, config)

    # Load LoRAs
    lora_ok = False
    if config.lora_paths:
        lora_ok = load_loras(pipe, config)

    # Apply CPU optimizations
    pipe = apply_cpu_optimizations(pipe, config)
    if refiner is not None:
        refiner = apply_cpu_optimizations(refiner, config)

    # Save prompt to history
    save_prompt_to_history(config.prompt, config.negative_prompt)

    # Generate
    start_time = time.time()
    try:
        if config.is_img2img:
            results = generate_img2img(pipe, config)
        else:
            results = generate_text2img(pipe, config, refiner=refiner)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            con.error("Out of memory! Try reducing image size or closing other applications.")
        else:
            con.error(f"Generation error: {exc}")
        results = []
    except Exception as exc:
        con.error(f"Unexpected error: {exc}")
        results = []

    elapsed = time.time() - start_time

    if results:
        con.success(f"Generated {len(results)} image(s) in {elapsed:.1f}s")
        if config.save_report:
            for img, path in results:
                generate_report(config, path, elapsed)
    else:
        con.error("No images were generated")

    # Cleanup
    cleanup_pipeline(pipe)
    if refiner is not None:
        cleanup_pipeline(refiner)
    gc.collect()

    return 0 if results else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the 'info' subcommand."""
    con.section("System Information")

    rows = [
        ["Python", sys.version.split()[0]],
        ["PyTorch", torch.__version__],
        ["Diffusers", diffusers.__version__],
        ["CPU Cores", str(os.cpu_count())],
        ["Device", "CPU"],
    ]

    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        rows.append(["Total RAM", f"{vm.total / (1024**3):.1f} GB"])
        rows.append(["Available RAM", f"{vm.available / (1024**3):.1f} GB"])
        rows.append(["RAM Usage", f"{vm.percent:.1f}%"])
        rows.append(["Process RAM", f"{get_memory_usage():.2f} GB"])
    else:
        rows.append(["psutil", "not installed"])

    con.table(["Property", "Value"], rows, title="System Info")

    if args.model:
        con.section("Model Detection")
        mtype = detect_model_type(args.model)
        con.info(f"Path: {args.model}")
        con.info(f"Detected type: {'SDXL' if mtype == 'sdxl' else 'SD 1.5'}")

        if os.path.exists(args.model):
            size_mb = os.path.getsize(args.model) / (1024 * 1024)
            con.info(f"File size: {size_mb:.0f} MB")
        else:
            con.warning("File not found — detection based on filename only")

    return 0


def cmd_schedulers() -> int:
    """Handle the 'schedulers' subcommand."""
    con.section("Available Schedulers")
    rows = []
    for key, info in SCHEDULER_REGISTRY.items():
        lcm_tag = " (LCM)" if info["lcm_ok"] else ""
        rows.append([
            key,
            info["label"] + lcm_tag,
            str(info["default_cfg"]),
            str(SCHEDULER_DEFAULT_STEPS.get(key, 20)),
        ])
    con.table(["Key", "Name", "Default CFG", "Default Steps"], rows, title="Schedulers")
    return 0


def cmd_presets() -> int:
    """Handle the 'presets' subcommand."""
    con.section("Available Presets")
    rows = []
    for key, p in PRESETS.items():
        rows.append([
            key,
            str(p["cfg_scale"]),
            str(p["steps"]),
            p["scheduler"],
            p["prompt_suffix"][:40] + "...",
        ])
    con.table(["Preset", "CFG", "Steps", "Scheduler", "Prompt Suffix"], rows, title="Presets")
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Handle the 'interactive' subcommand — full menu-driven UI."""
    con.banner(
        f"CPU-Optimized Stable Diffusion Pipeline v{VERSION}\n"
        f"SD 1.5 & SDXL  |  Multiple LoRA  |  13 Schedulers\n"
        f"{LICENSE_TEXT}"
    )

    pipe = None
    refiner = None
    config = None

    while True:
        con.section("Main Menu")
        con.print("  1. Text-to-Image")
        con.print("  2. Image-to-Image")
        con.print("  3. Load / Reload Model")
        con.print("  4. System Info")
        con.print("  5. List Schedulers")
        con.print("  6. List Presets")
        con.print("  7. Manage Configs")
        con.print("  8. Prompt History")
        con.print("  9. Exit")

        choice = con.prompt("Select", "1")

        if choice == "9":
            con.info("Goodbye!")
            if pipe is not None:
                cleanup_pipeline(pipe)
            if refiner is not None:
                cleanup_pipeline(refiner)
            break

        elif choice in ("1", "2"):
            # Build config interactively
            cfg = _interactive_build_config(is_img2img=(choice == "2"))

            if pipe is not None:
                cleanup_pipeline(pipe)
            if refiner is not None:
                cleanup_pipeline(refiner)
                refiner = None
            pipe = None

            cfg.resolve_defaults()

            # Load model
            try:
                pipe = load_pipeline(cfg)
            except Exception:
                con.error("Model loading failed. Returning to menu.")
                continue

            # Refiner
            if cfg._resolved_model_type == "sdxl" and cfg.refiner_path:
                refiner = load_refiner_pipeline(cfg.refiner_path)
            else:
                refiner = None

            # Warm up
            warm_up(pipe, cfg)

            # LoRA
            if cfg.lora_paths:
                load_loras(pipe, cfg)

            # Optimize
            pipe = apply_cpu_optimizations(pipe, cfg)
            if refiner is not None:
                refiner = apply_cpu_optimizations(refiner, cfg)

            # Generate
            start_time = time.time()
            try:
                if cfg.is_img2img:
                    results = generate_img2img(pipe, cfg)
                else:
                    results = generate_text2img(pipe, cfg, refiner=refiner)
            except Exception as exc:
                con.error(f"Generation failed: {exc}")
                results = []

            elapsed = time.time() - start_time

            if results:
                con.success(f"Generated {len(results)} image(s) in {elapsed:.1f}s")
                # Post-generation actions
                for img, path in results:
                    _post_generation_menu(img, path, cfg, elapsed)
            else:
                con.error("Generation failed")

            config = cfg

        elif choice == "3":
            model_path = con.prompt("Model path (file or HF repo ID)",
                                     getattr(args, "model", ""))
            if not model_path:
                con.warning("No model path provided")
                continue
            if pipe is not None:
                cleanup_pipeline(pipe)
                pipe = None
            if refiner is not None:
                cleanup_pipeline(refiner)
                refiner = None

            mtype = detect_model_type(model_path)
            con.info(f"Detected model type: {'SDXL' if mtype == 'sdxl' else 'SD 1.5'}")

            try:
                temp_cfg = GenerationConfig(model_path=model_path, model_type=mtype)
                temp_cfg.resolve_defaults()
                pipe = load_pipeline(temp_cfg)
                warm_up(pipe, temp_cfg)
                pipe = apply_cpu_optimizations(pipe, temp_cfg)
                config = temp_cfg
                con.success("Model loaded and ready")
            except Exception as exc:
                con.error(f"Failed to load model: {exc}")

        elif choice == "4":
            cmd_info(argparse.Namespace(model=""))

        elif choice == "5":
            cmd_schedulers()

        elif choice == "6":
            cmd_presets()

        elif choice == "7":
            _interactive_config_menu()

        elif choice == "8":
            _interactive_prompt_history()

        else:
            con.warning("Invalid choice")

    return 0


def _interactive_build_config(is_img2img: bool = False) -> GenerationConfig:
    """Build a GenerationConfig through interactive prompts."""
    con.section("Configuration")

    # Model
    model_path = con.prompt("Model path (.safetensors, .ckpt, or HF repo ID)")
    model_type = "auto"

    # Quick model type selection
    mtype_choice = con.prompt("Model type (1=Auto, 2=SD 1.5, 3=SDXL)", "1")
    if mtype_choice == "2":
        model_type = "sd15"
    elif mtype_choice == "3":
        model_type = "sdxl"

    # LoRA
    lora_paths: List[str] = []
    lora_weights: List[float] = []
    if con.confirm("Use LoRA?", default=False):
        num_loras_str = con.prompt("Number of LoRAs (1-5)", "1")
        try:
            num_loras = max(1, min(5, int(num_loras_str)))
        except ValueError:
            num_loras = 1

        for i in range(num_loras):
            path = con.prompt(f"LoRA #{i+1} path")
            if path:
                lora_paths.append(path)
                w_str = con.prompt(f"LoRA #{i+1} strength (0.0-2.0)", "1.0")
                try:
                    lora_weights.append(max(0.0, min(2.0, float(w_str))))
                except ValueError:
                    lora_weights.append(1.0)

    # Preset
    preset = ""
    if con.confirm("Apply a preset?", default=False):
        con.print("  Presets: " + ", ".join(PRESETS.keys()))
        preset = con.prompt("Preset name", "")
        if preset and preset not in PRESETS:
            con.warning(f"Unknown preset '{preset}', ignoring")
            preset = ""

    # Scheduler
    con.print("  Schedulers: " + ", ".join(SCHEDULER_REGISTRY.keys()))
    scheduler = con.prompt("Scheduler", "euler-a")
    if scheduler not in SCHEDULER_REGISTRY:
        con.warning(f"Unknown scheduler '{scheduler}', using euler-a")
        scheduler = "euler-a"

    # Steps
    default_steps = SCHEDULER_DEFAULT_STEPS.get(scheduler, 20)
    steps_str = con.prompt(f"Inference steps (default: {default_steps})", str(default_steps))
    try:
        steps = max(1, int(steps_str))
    except ValueError:
        steps = default_steps

    # CFG
    default_cfg = SCHEDULER_REGISTRY.get(scheduler, {}).get("default_cfg", 7.5)
    cfg_str = con.prompt(f"CFG scale (default: {default_cfg})", str(default_cfg))
    try:
        cfg_scale = max(0.0, min(30.0, float(cfg_str)))
    except ValueError:
        cfg_scale = default_cfg

    # Dimensions (for txt2img)
    width, height = 0, 0
    if not is_img2img:
        # Detect if SDXL for default
        is_sdxl = model_type == "sdxl" or (model_type == "auto" and detect_model_type(model_path) == "sdxl")
        dim_default = "1024x1024" if is_sdxl else "512x512"

        if con.confirm("Use aspect ratio?", default=False):
            ratios = ASPECT_RATIOS if is_sdxl else ASPECT_RATIOS_SD15
            con.print("  Ratios: " + ", ".join(ratios.keys()))
            ar = con.prompt("Aspect ratio (e.g. 16:9)", "1:1")
            if ar in ratios:
                width, height = ratios[ar]
            else:
                con.warning(f"Unknown ratio '{ar}', using default")
                width, height = (1024, 1024) if is_sdxl else (512, 512)
        else:
            dim_str = con.prompt(f"Dimensions WxH (default: {dim_default})", dim_default)
            try:
                parts = dim_str.lower().split("x")
                width = int(parts[0].strip())
                height = int(parts[1].strip())
            except Exception:
                width, height = (1024, 1024) if is_sdxl else (512, 512)

    # Prompt
    prompt = con.prompt("Prompt", "A beautiful landscape with mountains and a lake, ultra-detailed, realistic, 4k")
    neg_prompt = con.prompt("Negative prompt (optional)", "")

    # img2img specific
    image_path = ""
    strength = 0.75
    if is_img2img:
        image_path = con.prompt("Input image path")
        s_str = con.prompt("Transformation strength (0.1-1.0)", "0.75")
        try:
            strength = max(0.1, min(1.0, float(s_str)))
        except ValueError:
            strength = 0.75

    # SDXL Refiner
    refiner_path = ""
    refiner_strength = 0.3
    if model_type in ("sdxl", "auto"):
        if con.confirm("Use SDXL refiner?", default=False):
            refiner_path = con.prompt("Refiner model path")
            rs_str = con.prompt("Refiner strength (0.1-1.0)", "0.3")
            try:
                refiner_strength = max(0.1, min(1.0, float(rs_str)))
            except ValueError:
                refiner_strength = 0.3

    # Seed
    seed_str = con.prompt("Seed (-1 for random)", "-1")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = -1

    # Batch
    batch_str = con.prompt("Batch size (1-8)", "1")
    try:
        batch_size = max(1, min(8, int(batch_str)))
    except ValueError:
        batch_size = 1

    # Post-processing
    post_process = con.confirm("Apply post-processing?", default=False)

    # Output directory
    output_dir = con.prompt("Output directory", str(DEFAULT_OUTPUT_DIR))

    config = GenerationConfig(
        model_path=model_path,
        model_type=model_type,
        is_img2img=is_img2img,
        refiner_path=refiner_path,
        refiner_strength=refiner_strength,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        scheduler=scheduler,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        width=width,
        height=height,
        prompt=prompt,
        negative_prompt=neg_prompt,
        image_path=image_path,
        strength=strength,
        batch_size=batch_size,
        post_process=post_process,
        output_dir=output_dir,
    )

    # Apply preset overrides
    if preset and preset in PRESETS:
        p = PRESETS[preset]
        config.prompt += p["prompt_suffix"]
        if not config.negative_prompt:
            config.negative_prompt = p["negative_prompt"]

    # Save prompt to history
    save_prompt_to_history(config.prompt, config.negative_prompt)

    return config


def _post_generation_menu(image: Image.Image, output_path: str, config: GenerationConfig, elapsed: float) -> None:
    """Interactive post-generation action menu."""
    while True:
        con.section("Post-Generation Actions")
        con.print("  1. Open image")
        con.print("  2. Apply post-processing")
        con.print("  3. Save as JPEG")
        con.print("  4. Generate report")
        con.print("  5. Save config")
        con.print("  6. Generate again (same settings, new seed)")
        con.print("  7. Back to main menu")

        choice = con.prompt("Action", "7")

        if choice == "1":
            try:
                image.show()
            except Exception as exc:
                con.warning(f"Cannot open image viewer: {exc}")

        elif choice == "2":
            enhanced = apply_post_processing(image)
            ep = output_path.replace(".png", "_enhanced.png")
            safe_save_image(enhanced, ep)
            con.success(f"Enhanced image saved: {ep}")
            image = enhanced

        elif choice == "3":
            jp = output_path.replace(".png", ".jpg")
            safe_save_image(image, jp, quality=95)
            con.success(f"JPEG saved: {jp}")

        elif choice == "4":
            generate_report(config, output_path, elapsed)

        elif choice == "5":
            name = con.prompt("Config filename", f"config_{int(time.time())}.json")
            save_config(config, name)

        elif choice == "6":
            con.info("Regenerating...")
            # This will be handled by the caller's loop
            return

        elif choice == "7":
            return

        else:
            con.warning("Invalid choice")


def _interactive_config_menu() -> None:
    """Interactive config management menu."""
    con.section("Config Management")

    configs = list_configs()
    if configs:
        con.print("  Saved configs:")
        for i, c in enumerate(configs, 1):
            con.print(f"    {i}. {c}")
    else:
        con.info("No saved configs found")

    con.print("\n  1. Load config")
    con.print("  2. Delete config")
    con.print("  3. Back")

    choice = con.prompt("Action", "3")

    if choice == "1" and configs:
        idx_str = con.prompt("Config number", "1")
        try:
            idx = max(0, min(len(configs) - 1, int(idx_str) - 1))
            cfg = load_config(configs[idx])
            if cfg:
                con.success(f"Loaded: {configs[idx]}")
                _print_config_summary(cfg)
        except ValueError:
            con.warning("Invalid selection")

    elif choice == "2" and configs:
        idx_str = con.prompt("Config number to delete", "1")
        try:
            idx = max(0, min(len(configs) - 1, int(idx_str) - 1))
            path = CONFIG_DIR / configs[idx]
            os.remove(path)
            con.success(f"Deleted: {configs[idx]}")
        except (ValueError, OSError) as exc:
            con.warning(f"Failed: {exc}")


def _interactive_prompt_history() -> None:
    """Interactive prompt history browser."""
    con.section("Prompt History")
    history = load_prompt_history()

    if not history:
        con.info("No prompt history")
        return

    for i, item in enumerate(history[:15], 1):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(item["timestamp"]))
        preview = item["prompt"][:60] + ("..." if len(item["prompt"]) > 60 else "")
        con.print(f"  {i}. [{ts}] {preview}")

    con.print(f"\n  (Total: {len(history)} entries, showing latest 15)")


def _print_config_summary(config: GenerationConfig) -> None:
    """Print a summary table of the generation config."""
    is_sdxl = config._resolved_model_type == "sdxl"
    model_label = "SDXL" if is_sdxl else "SD 1.5"
    sched_label = SCHEDULER_REGISTRY.get(config.scheduler, {}).get("label", config.scheduler)

    rows = [
        ["Model", f"{os.path.basename(config.model_path)} ({model_label})"],
        ["Mode", "Image-to-Image" if config.is_img2img else "Text-to-Image"],
        ["Prompt", config.prompt[:70] + ("..." if len(config.prompt) > 70 else "")],
        ["Negative", config.negative_prompt[:50] or "(none)"],
        ["Scheduler", sched_label],
        ["Steps", str(config.steps)],
        ["CFG Scale", f"{config.cfg_scale:.1f}"],
        ["Seed", str(config._actual_seed)],
        ["Dimensions", f"{config.width}x{config.height}"],
        ["Batch", str(config.batch_size)],
    ]

    if config.is_img2img:
        rows.append(["Strength", f"{config.strength:.2f}"])
        rows.append(["Input Image", os.path.basename(config.image_path) or "(none)"])

    if config.lora_paths:
        for i, (p, w) in enumerate(zip(config.lora_paths, config.lora_weights)):
            rows.append([f"LoRA #{i+1}", f"{os.path.basename(p)} ({w:.2f})"])

    if config.refiner_path:
        rows.append(["Refiner", os.path.basename(config.refiner_path)])
        rows.append(["Refiner Str.", f"{config.refiner_strength:.2f}"])

    con.table(["Parameter", "Value"], rows, title="Generation Configuration")


#MAIN

def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # No subcommand --> launch interactive
    if args.command is None:
        # Print help if no args at all, or launch interactive
        if len(sys.argv) == 1:
            return cmd_interactive(argparse.Namespace(model=""))
        parser.print_help()
        return 0

    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "schedulers":
        return cmd_schedulers()
    elif args.command == "presets":
        return cmd_presets()
    elif args.command == "interactive":
        return cmd_interactive(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        con.info("\nInterrupted. Cleaning up...")
        gc.collect()
        con.info("Done.")
        sys.exit(130)
    except Exception as exc:
        con.error(f"Fatal error: {exc}")
        con.info("Possible causes: outdated diffusers, incompatible model, missing dependencies")
        con.info("Fix:  pip install --upgrade diffusers transformers accelerate safetensors")
        sys.exit(1)
