"""
rays.py
"""

from dataclasses import dataclass
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch


@dataclass(init=False)
class RayBundle:
    """
    Ray bundle class.
    """

    origins: Float[torch.Tensor, "*batch_size 3"]
    """Ray origins in the world coordinate."""
    directions: Float[torch.Tensor, "*batch_size 3"]
    """Ray directions in the world coordinate."""
    nears: Float[torch.Tensor, "*batch_size 1"]
    """Near clipping plane."""
    fars: Float[torch.Tensor, "*batch_size 1"]
    """Far clipping plane."""

    def __init__(
        self,
        origins: Float[torch.Tensor, "*batch_size 3"],
        directions: Float[torch.Tensor, "*batch_size 3"],
        nears: Float[torch.Tensor, "*batch_size 1"],
        fars: Float[torch.Tensor, "*batch_size 1"],
    ) -> None:
        """
        Initializes RayBundle.
        """
        self.origins = origins
        self.directions = directions
        self.nears = nears
        self.fars = fars

    def __len__(self) -> int:
        """
        Returns the number of rays in the bundle.
        """
        return self.origins.shape[0]

@dataclass(init=False)
class RaySamples:
    """
    Ray sample class.
    """

    ray_bundle: RayBundle
    """Ray bundle. Contains ray origin, direction, near and far bounds."""
    t_samples: Float[torch.Tensor, "num_ray num_sample"]
    """Distance values sampled along rays."""

    def __init__(
        self,
        ray_bundle: RayBundle,
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
    ) -> None:
        """
        Initializes RaySample.
        """
        self.ray_bundle = ray_bundle
        self.t_samples = t_samples

    @jaxtyped
    @typechecked
    def compute_sample_coordinates(self) -> Float[torch.Tensor, "num_ray num_sample 3"]:
        """
        Computes coordinates of points sampled along rays in the ray bundle.

        Returns:
            coords: Coordinates of points sampled along rays in the ray bundle.
        """
        # TODO
        origins = self.ray_bundle.origins
        origins = origins.unsqueeze(1).repeat(1, self.t_samples.shape[1], 1) #expand to num_sample
        directions = self.ray_bundle.directions
        directions = directions.unsqueeze(1).repeat(1, self.t_samples.shape[1], 1) #expand to num_sample
        t_samples = self.t_samples.unsqueeze(-1)

        sample_coords = origins + t_samples * directions

        return sample_coords


    @jaxtyped
    @typechecked
    def compute_deltas(self, right_end: float=1e8) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Compute differences between adjacent t's required to approximate integrals.

        Args:
            right_end: The value to be appended to the right end
                when computing 1st order difference.

        Returns:
            deltas: Differences between adjacent t's.
                When evaluating the delta for the farthest sample on a ray,
                use the value of the argument 'right_end'.
        """
        # TODO
        # HINT: Look up to the documentation of torch.diff.
        #compute t_(i+1) - t_i 
        t_samples = self.t_samples
        deltas = torch.diff(
            torch.cat((t_samples, right_end * torch.ones((t_samples.shape[0], 1), device=t_samples.device)), 
            dim=-1
            ) #add right end to the end of t_samples
        )

        return deltas