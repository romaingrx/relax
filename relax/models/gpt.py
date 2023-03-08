#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 February 03, 18:14:12
@last modified : 2023 March 09, 00:07:12
"""

import jax
import math
import einops
import haiku as hk
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from relax.models import CausalSelfAttention

from typing import Optional

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


@dataclass
class MLP(hk.Module):
    n_embed : int
    dropout_rate: float
    bias: Optional[bool] = False
    name: Optional[str] = "MLP"

    def __call__(self, x, training: bool = True):
        x = hk.Linear(4 * self.n_embed, with_bias=self.bias, name='c_fc')(x)
        x = new_gelu(x)
        x = hk.Linear(self.n_embed, with_bias=self.bias, name='c_proj')(x)
        if training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        return x

@dataclass
class GPTBlock(hk.Module):
    n_embed : int
    n_head : int
    dropout_rate : float 
    bias : Optional[bool] = False
    name : str = "GPTBlock" 
    
    def __call__(self, x, training: bool = True):
        x = hk.LayerNorm(create_scale=True, create_offset=self.bias, axis=-1, name='ln_1')(x) # Normalize the batch
        x = CausalSelfAttention(self.n_head, self.n_embed, name="attn")(x).projection # Apply the self causal attention
        x += x # Skip connection
        x = hk.LayerNorm(create_scale=True, create_offset=self.bias, axis=-1, name='ln_2')(x) # Normalize the skipped connection
        x = MLP(self.n_embed, self.dropout_rate, self.bias, name='mlp')(x, training) # Apply the MLP
        x += x 
        return x

@dataclass
class GPT(hk.Module):
    vocab_size : int
    block_size : int
    n_blocks : int
    n_embed : int
    n_head : int 
    dropout_rate : int
    bias : Optional[bool] = False
    name : Optional[str] = "GPT"

    def __call__(self, idx, training: bool = True):
        b, t = idx.shape
        assert t <= self.block_size, f"Sequence of length {t} is too long for the block size {self.block_size}"

        # If there is an error, maybe adapt the type to long (2^63 - 1)
        pos = jnp.arange(0, t)[None]

        # Embed the tokens indices in the \mathbb{R}^{embed_dim} space
        tok_emb = hk.Embed(self.vocab_size, self.n_embed, name='wte')(idx)
        # Embed the positions to match the embedding space (Also possible to embed without trainable parameters)
        pos_emb = hk.Embed(self.block_size, self.n_embed, name='wpe')(pos)

        x = pos_emb + tok_emb
        if training:
            # Add the embeddings and apply a dropout on them
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        for i in range(self.n_blocks):
            x = GPTBlock(self.n_embed, self.n_head, self.dropout_rate, self.bias, name=f"block_{i}")(x, training)
        x = hk.LayerNorm(create_scale=True, create_offset=self.bias, axis=-1, name='ln_f')(x) 
        # There is a possibility to speed-up the inference time by just applying the lm_head on the last position
        logits = hk.Linear(self.vocab_size, with_bias=False, name="lm_head")(x) 
        return logits
