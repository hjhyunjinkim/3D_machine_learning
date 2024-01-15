import torch
from torchvision import transforms
import PIL.Image
from ddpm import DiffusionModule
from scheduler import DDIMScheduler, DDPMScheduler
from network import UNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import argparse
from dotmap import DotMap

def create_mask(mask_size=32):
    """
    create a torch.Tensor mask with size=mask_size in the center of a 256 x 256 image.
    input : mask_size : int
    output : torch.Tensor (with only 0s and 1s)
    1 - no mask
    0 - mask
    """
    if mask_size % 2 == 0:
        mask_size += 1
    
    #create an white image array
    mask = torch.ones((256, 256))

    start = (256 - mask_size) // 2
    end_point = start + mask_size
    mask[start:end_point, start:end_point] = 0

    return mask

def repaint(image: torch.Tensor, mask: torch.Tensor, model: DiffusionModule, scheduler: DDIMScheduler, num_samples: int = 1, return_traj: bool = False):
    """
    Repaint an image using a learned diffusion model.
    """
    assert image.shape == mask.shape
    assert image.shape[0] == 1
    assert image.shape[1] == 3
    assert image.shape[2] == image.shape[3]

    image_resolution = image.shape[2]
    image = image.to(model.device)
    mask = mask.to(model.device)

    image = image * mask
    image = image + (1 - mask) * torch.randn_like(image) * 0.01

    traj = [image]
    for t in scheduler.timesteps:
        x_t = traj[-1]
        noise_pred = model.network(x_t, timestep=t.to(model.device))

        x_t_prev = scheduler.step(x_t, t, noise_pred)

        traj[-1] = traj[-1].cpu()
        traj.append(x_t_prev.detach())

    if return_traj:
        return traj
    else:
        return traj[-1]
    

def main(args):
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"
    image_resolution = config.image_resolution
    # Load a trained model.
    


    #Transform data
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
        transforms.Lambda(lambda t : t.unsqueeze(0))
    ])
    #create mask
    mask = create_mask(32)

    image = PIL.Image.open("image.png") #change to real path
    
    #bring model and ddim
    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
    )

    ddpm = DiffusionModule(network, None)

    #repaint and show result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=-1,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--use_cfg", type=bool, default=True)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
