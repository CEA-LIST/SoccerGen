import torch
import torchvision
import numpy as np
import einops
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='Configuration file path.')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the image.')
parser.add_argument('--batch_size', type=int, default=4, help='The number of images to sample.')
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--ddim_eta', type=float, default=0.)
parser.add_argument('--cfg_scale', type=float, default=9.0)
parser.add_argument('--cfg_rw', action="store_true", help="Whether to add classifier-free guidance reweighting.")
parser.add_argument('--negative_prompts', type=str, default="lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
parser.add_argument('--save_path', type=str, default=None, help="Path to the folder where to save the images.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = OmegaConf.load(args.config)
model = instantiate_from_config(config.model)
if args.cfg_rw:
    model.control_scales = [args.cfg_scale * (0.825 ** float(12 - i)) for i in range(13)]

if args.ckpt_path is not None:
    model.init_from_ckpt(args.ckpt_path)

model = model.to(device)

prompt = "A TV broacast image of a professional soccer game. Players  of the two teams are scattered around the field, some running, some kicking the ball. White lines mark the boundaries of the field. Players of the same team have the same jersey and the two teams' jerseys are different."
text_embedding = model.get_learned_conditioning([prompt]*args.batch_size)

c = dict(c_crossattn=[text_embedding])


use_ddim = args.ddim_steps is not None
samples, _ = model.sample_log(cond=c, batch_size=args.batch_size, ddim=use_ddim, ddim_steps=args.ddim_steps, eta=args.ddim_eta)
images = model.decode_first_stage(samples).detach().cpu()
images = torch.clamp(images, -1., 1.)

images_cfg = None
if args.cfg_scale > 1.0:
    negative_prompt = args.negative_prompts
    uc_cross = model.get_learned_conditioning([negative_prompt]*args.batch_size)
    uc_full = {"c_crossattn": [uc_cross]}
    samples_cfg, _ = model.sample_log(
        cond=c,
        batch_size=args.batch_size, ddim=use_ddim,
        ddim_steps=args.ddim_steps, eta=args.ddim_eta,
        unconditional_guidance_scale=args.cfg_scale,
        unconditional_conditioning=uc_full,
    )
    images_cfg = model.decode_first_stage(samples_cfg).detach().cpu()
    images_cfg = torch.clamp(images_cfg, -1., 1.)


grid = torchvision.utils.make_grid(images, nrow=args.batch_size)
grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
grid = grid.numpy()
grid = (grid * 255).astype(np.uint8)
grid_image = Image.fromarray(grid)

dest = Path("." if args.save_path is None else args.save_path)
if not dest.exists():
    dest.mkdir(parents=True)

samples_path = dest / 'samples.jpg'
grid_image.save(samples_path)

if images_cfg is not None:
    grid_cfg = torchvision.utils.make_grid(images_cfg, nrow=args.batch_size)
    grid_cfg = (grid_cfg + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid_cfg = grid_cfg.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid_cfg = grid_cfg.numpy()
    grid_cfg = (grid_cfg * 255).astype(np.uint8)
    grid_image_cfg = Image.fromarray(grid_cfg)
    samples_cfg_path = dest / f'samples_cfg{args.cfg_scale}.jpg'
    grid_image_cfg.save(samples_cfg_path)
