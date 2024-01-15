"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim

        #NeRF Architecture -- DONE
        self.fn_in = nn.Linear(self.pos_dim, self.feat_dim)
        self.fn_1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_3 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_4 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_5 = nn.Linear(self.feat_dim + self.pos_dim, self.feat_dim)
        self.fn_6 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_7 = nn.Linear(self.feat_dim, self.feat_dim)
        self.fn_8 = nn.Linear(self.feat_dim, self.feat_dim + 1) #add volume density
        self.fn_9 = nn.Linear(self.feat_dim + self.view_dir_dim, self.feat_dim // 2)
        self.fn_out = nn.Linear(self.feat_dim // 2, 3) #rgb = 3

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        x = self.relu(self.fn_in(pos))
        x = self.relu(self.fn_1(x))
        x = self.relu(self.fn_2(x))
        x = self.relu(self.fn_3(x))
        x = self.relu(self.fn_4(x))
        x = torch.cat((x, pos), dim=-1) #skip connection - concatenates input

        x = self.relu(self.fn_5(x))
        x = self.relu(self.fn_6(x))
        x = self.relu(self.fn_7(x))
        x = self.fn_8(x)

        sigma = self.relu(x[:, 0]) #first column of x is sigma
        sigma = sigma.unsqueeze(1) #add dimension for broadcasting
        #DELETE FIRST COLUMN!!!!! - Sigma
        x = torch.cat((x[:, 1:], view_dir), dim=-1) #skip connection - concatenates view direction
        x = self.relu(self.fn_9(x))
        rgb = self.sigmoid(self.fn_out(x))

        return sigma, rgb
