import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jaxtyping import Int, Float


class PositionalEncoding(nn.Module):

    def __init__(self, t_channel: Int):
        """
        (Optional) Initialize positional encoding network

        Args:
            t_channel: number of modulation channel
        """
        super().__init__()
        self.t_channel = t_channel

    def forward(self, t):
        """
        Return the positional encoding of

        Args:
            t: input time

        Returns:
            emb: time embedding
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = []
        for i in range(self.t_channel):
            emb.append(torch.sin(2**i * t))
            emb.append(torch.cos(2**i * t))
        emb = torch.cat(emb, dim=-1)
        return emb


class MLP(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 out_dim: Int,
                 hid_shapes: Int[torch.Tensor, '...']):
        '''
        (TODO) Build simple MLP -- CHANGE 

        Args:
            in_dim: input dimension
            out_dim: output dimension
            hid_shapes: array of hidden layers' dimension
        '''
        super().__init__()
        shapes = [in_dim] + hid_shapes
        layers = []
        for i in range(len(shapes)-1):
            layers.append(nn.Linear(shapes[i], shapes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(shapes[-1], out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SimpleNet(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 enc_shapes: Int[torch.Tensor, '...'],
                 dec_shapes: Int[torch.Tensor, '...'],
                 z_dim: Int):
        super().__init__()
        '''
        (TODO) Build Score Estimation network. -- DONE
        You are free to modify this function signature.
        You can design whatever architecture.

        hint: it's recommended to first encode the time and x to get
        time and x embeddings then concatenate them before feeding it
        to the decoder.

        Args:
            in_dim: dimension of input
            enc_shapes: array of dimensions of encoder
            dec_shapes: array of dimensions of decoder
            z_dim: output dimension of encoder
        '''
        
        self.t_encoder = nn.Sequential(
            PositionalEncoding(10), #t_channel = 10
            MLP(20, z_dim, enc_shapes), # 10 * 2 - sin and cos
            nn.ReLU()
        )   
        self.x_encoder = nn.Sequential(
            MLP(in_dim, z_dim, enc_shapes),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            MLP(z_dim*2, in_dim, dec_shapes),
        )


    def forward(self, t, x):
        '''
        (TODO) Implement the forward pass. This should output
        the score s of the noisy input x.

        hint: you are free

        Args:
            t: the time that the forward diffusion has been running
            x: the noisy data after t period diffusion
        '''
        t_emb = self.t_encoder(t)
        z = self.x_encoder(x)
        z = torch.cat([z, t_emb], dim=-1)
        s = self.decoder(z)

        return s

