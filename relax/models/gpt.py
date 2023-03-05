#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 February 03, 18:14:12
@last modified : 2023 March 03, 16:39:00
"""

import jax
import einops
import haiku as hk
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from relax.models import CausalSelfAttention

from typing import Optional


@dataclass
class MLP(hk.Module):
    n_embed : int
    dropout_rate: float
    name: Optional[str] = "MLP"

    def __call__(self, x):
        x = hk.Linear(4 * self.n_embed)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self.n_embed)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        return x

@dataclass
class GPTBlock(hk.Module):
    n_embed : int
    n_head : int
    dropout_rate : float 
    name : str = "GPTBlock" 
    
    def __call__(self, x):
        x = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x) # Normalize the batch
        x = CausalSelfAttention(self.n_head, self.n_embed, name="SelfCausalAttention")(x).projection # Apply the self causal attention
        x += x # Skip connection
        x = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x) # Normalize the skipped connection
        x = MLP(self.n_embed, self.dropout_rate)(x)
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
    name : str = "GPT"

    def __call__(self, idx):
        b, t = idx.shape
        assert t <= self.block_size, f"Sequence of length {t} is too long for the block size {self.block_size}"

        # If there is an error, maybe adapt the type to long (2^63 - 1)
        pos = jnp.arange(0, t)[None]

        # Embed the tokens indices in the \mathbb{R}^{embed_dim} space
        tok_emb = hk.Embed(self.vocab_size, self.n_embed)(idx)
        # Embed the positions to match the embedding space (Also possible to embed without trainable parameters)
        pos_emb = hk.Embed(self.block_size, self.n_embed)(pos)

        # Add the embeddings and apply a dropout on them
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, pos_emb + tok_emb)
        for i in range(self.n_blocks):
            x = GPTBlock(self.n_embed, self.n_head, self.dropout_rate, name=f"GPTBlock_{i}")(x)
        x = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x) 
        # There is a possibility to speed-up the inference time by just applying the lm_head on the last position
        logits = hk.Linear(self.vocab_size, name="lm_head")(x) 
        return logits
