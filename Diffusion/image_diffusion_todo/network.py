from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(
        self,
        T: int = 1000,
        image_resolution: int = 64,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 2],
        attn: List[int] = [1],
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        # You can use the modules in `module.py`.
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(tdim)

        # TODO: classifier-free guidance -- DONE
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None
            self.num_classes = num_classes
            self.class_embedding = nn.Embedding(self.num_classes, tdim)
            
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()
        
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, class_label=None):
        """
        Input:
            x (`torch.Tensor [B,C,H,W]`)
            timestep (`torch.Tensor [B]`)
            class_label (`torch.Tensor [B]`, optional)
        Output:
            out (`torch.Tensor [B,C,H,W]`): noise prediction.
        """
        assert (
            x.shape[-1] == x.shape[-2] == self.image_resolution
        ), f"The resolution of x ({x.shape[-2]}, {x.shape[-1]}) does not match with the image resolution ({self.image_resolution})."
        # Timestep embedding
        temb = self.time_embedding(timestep)

        if self.use_cfg and class_label is not None:
            #TODO -- Classifier Guidance -- DONE 
            class_emb = self.class_embedding(class_label)
            temb = temb + class_emb   
            #temb = nn.functional.dropout(temb, p=self.cfg_dropout, training=self.training)

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h