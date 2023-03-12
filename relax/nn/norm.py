#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 March 10, 13:17:41
@last modified : 2023 March 10, 13:29:55
"""

import haiku as hk
import jax.numpy as jnp
from typing import Optional, Union, Sequence
from dataclasses import dataclass

@dataclass
class LayerNorm(hk.Module):
    ndim : Union[int, Sequence[int]]
    eps : Optional[float] = 1e-5
    gamma : Optional[bool] = True
    beta : Optional[bool] = True 
    name : Optional[str] = 'ln'
    
    def __call__(self, x):
        ndim = (self.ndim,) if isinstance(self.ndim, int) else self.ndim
        γ = hk.get_parameter('scale', ndim, init=jnp.ones) if self.gamma else jnp.ones(self.ndim)
        β = hk.get_parameter('offset', ndim, init=jnp.zeros) if self.beta else jnp.zeros(self.ndim)
        return γ * (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + self.eps) + β
