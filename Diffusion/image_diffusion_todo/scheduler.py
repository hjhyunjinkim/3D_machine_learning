from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        # TODO: Compute alphas and alphas_cumprod
        # alphas and alphas_cumprod correspond to $\alpha$ and $\bar{\alpha}$ in the DDPM paper (https://arxiv.org/abs/2006.11239).
        alphas = alphas_cumprod = betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor(1.0), self.alphas_cumprod[-1:]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, sample: torch.Tensor, timestep: int, noise_pred: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            timestep (`int`): current timestep in a reverse process.
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        # TODO: Implement the DDPM's one step denoising function.
        # Refer to Algorithm 2 in the DDPM paper (https://arxiv.org/abs/2006.11239).

        sample_prev = sample

        return sample_prev

    def add_noise(
        self,
        original_sample: torch.Tensor,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            timesteps: (`torch.IntTensor [B]`)
            noise: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_noisy: (`torch.Tensor [B,C,H,W]`): noisy samples
            noise: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        # TODO: Implement the function that samples $\mathbf{x}_t$ from $\mathbf{x}_0$.
        # Refer to Equation 4 in the DDPM paper (https://arxiv.org/abs/2006.11239).

        noisy_sample = noise = original_sample

        return noisy_sample, noise


class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the timesteps of a diffusion Markov chain. It is for accelerated generation process (Sec. 4.2) in the DDIM paper (https://arxiv.org/abs/2010.02502).
        """
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        sample: torch.Tensor,
        timestep: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ):
        """
        One step denoising function of DDIM: $x_{\tau_i}$ -> $x_{\tau_{i-1}}$.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep $\tau_i$.
            timestep (`int`): current timestep in a reverse process.
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        # TODO: Implement the DDIM's one step denoising function.
        # Refer to Equation 12 in the DDIM paper (https://arxiv.org/abs/2010.02502).

        sample_prev = sample

        return sample_prev
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        # TODO: Compute alphas and alphas_cumprod -- DONE
        # alphas and alphas_cumprod correspond to $\alpha$ and $\bar{\alpha}$ in the DDPM paper (https://arxiv.org/abs/2006.11239).
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

    def add_noise(
        self,
        original_sample: torch.Tensor,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0)

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            timesteps: (`torch.IntTensor [B]`)
            noise: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_noisy: (`torch.Tensor [B,C,H,W]`): noisy samples
            noise: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        # TODO: Implement the function that samples $\mathbf{x}_t$ from $\mathbf{x}_0$. -- DONE
        # Refer to Equation 4 in the DDPM paper (https://arxiv.org/abs/2006.11239).

        if noise is None:
            noise = torch.randn_like(original_sample)
        
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps]**0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        one_minus_alpha_cumprod = 1 - self.alphas_cumprod[timesteps]  #Maybe need sqrt???? - CHECK 
        one_minus_alpha_cumprod = one_minus_alpha_cumprod.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noisy_sample = sqrt_alpha_cumprod * original_sample + one_minus_alpha_cumprod * noise

        return noisy_sample, noise


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("alphas_cumprod_t_prev", alphas_cumprod_t_prev)
        self.register_buffer("sigmas", sigmas)

    def step(self, sample: torch.Tensor, timestep: int, noise_pred: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t. (= x_t)
            timestep (`int`): current timestep in a reverse process. (t)
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model. (= \tilde{\epsilon}_t)
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        # TODO: Implement the DDPM's one step denoising function. -- DONE
        # Refer to Algorithm 2 in the DDPM paper (https://arxiv.org/abs/2006.11239).
        if timestep > 1:
            z = torch.randn_like(sample)  # z ~ N(0, I)
        else:
            z = 0

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_t_prev[timestep]
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        beta_t = 1 - alpha_t
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        #ORIGINAL EXAMPLE
        x_0 = (sample - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5

        original_coeff = alpha_prod_t_prev**0.5 * beta_t / beta_prod_t
        current_coeff = alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = original_coeff * x_0 + current_coeff * sample
        
        #NOISE ADDITION -- TODO
        var = 0
        if timestep > 0:
            z = torch.randn(sample.shape, device=sample.device)
            var = self.sigmas[timestep] * z

        sample_prev = pred_prev_sample + var
 
        return sample_prev


class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the timesteps of a diffusion Markov chain. It is for accelerated generation process (Sec. 4.2) in the DDIM paper (https://arxiv.org/abs/2010.02502).
        """
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        self.step_ratio = step_ratio
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        sample: torch.Tensor,
        timestep: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ):
        """
        One step denoising function of DDIM: $x_{\tau_i}$ -> $x_{\tau_{i-1}}$.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep $\tau_i$. (= $x_{\tau_i}$)
            timestep (`int`): current timestep in a reverse process. ($\tau_i$)
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model. 
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        # TODO: Implement the DDIM's one step denoising function. -- DONE
        # Refer to Equation 12 in the DDIM paper (https://arxiv.org/abs/2010.02502).
        final_alpha_cumprod = torch.tensor(1.0)
        prev_timestep = timestep - self.step_ratio

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        #predicted x_0
        x_0 = (sample - (beta_prod_t**0.5) * noise_pred) / alpha_prod_t**0.5

        #predict random noise
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        sigma_t = eta * variance**0.5

        #direction pointing to x_t
        pred_direction = (1 - alpha_prod_t_prev - sigma_t**2)**0.5 * noise_pred

        sample_prev = alpha_prod_t_prev**0.5 * x_0 + pred_direction + sigma_t * torch.randn(sample.shape, device=sample.device)

        return sample_prev
