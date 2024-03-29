import abc
import torch
import numpy as np
#from jaxtyping import Array
from jaxtyping import Float

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of discretization steps
        self.T = T         # terminal time
        self.dt = T / N
        self.is_reverse = False
        self.is_bridge = False

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def predict_fn(self, x):
        return NotImplemented

    @abc.abstractmethod
    def correct_fn(self, t, x):
        return NotImplemented

    def dw(self, x, dt=None):
        """
        (TODO) Return the differential of Brownian motion -- DONE

        Args:
            x: input data

        Returns:
            dw (same shape as x)
        """
        dt = self.dt if dt is None else dt
        #shape = x.shape
        #dw = np.sqrt(dt) * torch.randn(shape)
        dw = torch.randn_like(x) * (dt**0.5)

        return dw

    def prior_sampling(self, x):
        """
        Sampling from prior distribution. Default to unit gaussian.

        Args:
            x: input data

        Returns:
            z: random variable with same shape as x
        """
        return torch.randn_like(x)

    def predict_fn(self,
                   t,
                   x,
                   dt: Float=None):
        """
        (TODO) Perform single step diffusion. -- DONE

        Args:
            t: current diffusion time
            x: input with noise level at time t (x_t)
            dt: the discrete time step. Default to T/N

        Returns:
            x: input at time t+dt
        """
        dt = self.dt if dt is None else dt # default to T/N
        drift, diffusion = self.sde_coeff(t, x)
        pred = x + drift * dt + diffusion * self.dw(x, dt)
        return pred

    def correct_fn(self, t, x):
        return None

    def reverse(self, model):
        N = self.N
        T = self.T
        forward_sde_coeff = self.sde_coeff

        class RSDE(self.__class__):
            def __init__(self, score_fn):
                super().__init__(N, T)
                self.score_fn = score_fn
                self.is_reverse = True
                self.forward_sde_coeff = forward_sde_coeff

            def sde_coeff(self, t, x):
                """
                (TODO) Return the reverse drift and diffusion terms. -- DONE

                Args:
                    t: current diffusion time
                    x: current input at time t

                Returns:
                    reverse_f: reverse drift term
                    g: reverse diffusion term
                """
                score = self.score_fn(T-t, x)
                drift, diffusion = self.forward_sde_coeff(T-t, x)
                reverse_f = drift - (diffusion**2) * score
                g = diffusion
                return reverse_f, g

            def ode_coeff(self, t, x):
                """
                (Optional) Return the reverse drift and diffusion terms in
                ODE sampling.

                Args:
                    t: current diffusion time
                    x: current input at time t

                Returns:
                    reverse_f: reverse drift term
                    g: reverse diffusion term
                """
                reverse_f = None
                g         = None
                return reverse_f, g

            def predict_fn(self,
                           t,
                           x,
                           dt=None,
                           ode=False):
                """
                (TODO) Perform single step reverse diffusion -- DONE
                First, consider only when ode is False
                """
                dt = self.dt if dt is None else dt
                reverse_f, g = self.sde_coeff(t, x)
                #pred = x + reverse_f * dt + g * self.dw(x, dt)
                pred = x - reverse_f * dt + g * self.dw(x, dt)
                return pred

        return RSDE(model)

class OU(SDE):
    def __init__(self, N=1000, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        f = -0.5 * x
        g = torch.ones(x.shape)
        return f, g

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.5 * t
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1 - torch.exp((2 * log_mean_coeff)[:,None])) * torch.ones_like(x)
        return mean, std

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    def sde_coeff(self, t, x):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        f = torch.zeros_like(x)
        log_mean_coeff = torch.tensor(np.log(self.sigma_max) - np.log(self.sigma_min))[:, None]
        g = sigma * torch.sqrt(2 * log_mean_coeff) * torch.ones_like(x)
        return f, g

    def marginal_prob(self, t, x):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t[:, None]
        mean = x
        return mean, std


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def sde_coeff(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        f = -0.5 * beta_t[:, None] * x
        g = torch.sqrt(beta_t[:, None])
        return f, g

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff[:, None])) * torch.ones_like(x)
        return mean, std


class SB(abc.ABC):
    def __init__(self, N=1000, T=1, zf_model=None, zb_model=None):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time
        self.dt = T / N

        self.is_reverse = False
        self.is_bridge  = True

        self.zf_model = zf_model
        self.zb_model = zb_model

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x) * (dt**0.5)

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    def sb_coeff(self, t, x):
        """
        (Optional) Return the SB reverse drift and diffusion terms.

        Args:
        """
        sb_f = None
        g    = None
        return sb_f, g

    def predict_fn(self,
                   t,
                   x,
                   dt:Float =None):
        """
        Args:
            t:
            x:
            dt:
        """
        return x

    def correct_fn(self, t, x, dt=None):
        return x

    def reverse(self, model):
        """
        (Optional) Initialize the reverse process
        """

        class RSB(self.__class__):
            def __init__(self, model):
                super().__init__(N, T, zf_model, zb_model)
                """
                (Optional) Initialize the reverse process
                """

            def sb_coeff(self, t, x):
                """
                (Optional) Return the SB reverse drift and diffusion terms.
                """
                sb_f = None
                g    = None
                return sb_f, g

        return RSDE(model)

class OUSB(SB):
    def __init__(self, N=1000, T=1, zf_model=None, zb_model=None):
        super().__init__(N, T, zf_model, zb_model)

    def sde_coeff(self, t, x):
        f = -0.5 * x
        g = torch.ones(x.shape)
        return f, g