import argparse
import os
import sys
import glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from datauri import DataURI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

def get_device():
    if (torch.cuda.is_available()):
        return 'cuda'
    elif (torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'

from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


config = 'configs/stable-diffusion/v1-inference.yaml'
ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
device = torch.device(get_device())
model = model.to(device)


def generateImage(
    prompt,
    seed,
    width,
    height,
    ddim_steps,
):
    outdir = 'outputs/images'
    skip_grid = False
    skip_save = False
    ddim_steps = 50
    plms = True
    fixed_code = False
    ddim_eta = 0.0
    n_iter = 1
    H = width or 512
    W = height or 768
    C = 4
    f = 8
    n_samples = 1
    n_rows = 0
    scale = 7.5
    from_file = None
    # config = 'configs/stable-diffusion/v1-inference.yaml'
    # ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'
    precision = 'autocast'
    results = []

    seed_everything(seed)

    # config = OmegaConf.load(f"{config}")
    # model = load_model_from_config(config, f"{ckpt}")

    # device = torch.device(get_device())
    # model = model.to(device)
    sampler = PLMSSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "ScrollRack"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if fixed_code:
        start_code = torch.randn(
            [n_samples, C, H // f, W // f], device="cpu"
        ).to(torch.device(device))

    precision_scope = autocast if precision == "autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext  # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(
                                batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_image_torch = torch.from_numpy(
                            x_samples_ddim).permute(0, 3, 1, 2)

                        if not skip_save:
                            for x_sample in x_image_torch:
                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(
                                    sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not skip_grid:
                            all_samples.append(x_image_torch)

                if not skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * \
                        rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    filepath = os.path.join(outpath, f'grid-{grid_count:04}.png')

                    img.save(filepath)
                    results.append(DataURI.from_file(filepath))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    return results, seed
