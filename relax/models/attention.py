#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 November 11, 17:43:12
@last modified : 2023 May 26, 10:21:33
"""

import jax
import jax.numpy as jnp
import haiku as hk
import einops

from relax.utils import AttrDict

from typing import Optional
from dataclasses import dataclass

# TODO: create a faster operation for dot product attention (see deepspeed or the latest pytorch 2.0 implementation)

@dataclass
class MultiHeadAttention(hk.Module):
    num_heads : int # The number of heads to divide the embed_dim into
    embed_dim : int # The embedding dimension (d_model in the paper)
    k_dim : Optional[int] = None # Dimension of the key projected space (default: embed_dim)
    v_dim : Optional[int] = None # Dimension of the value projected space (default: embed_dim)
    bias : Optional[bool] = False # Whether to use a bias in the linear layers
    dropout : Optional[float] = 0.0 # Dropout regularization
    rotary_encoding : Optional[bool] = False # Whether to use rotary encoding
    name : Optional[str] = None
    
    def __call__(self, q, k, v, mask=None, training=False, return_attention=True):
        assert self.embed_dim % self.num_heads == 0, "The embed dimension should be divisible by the number of heads."
        # Get a linear layer for the q, k, v at the same time
        Q = hk.Linear(self.k_dim or self.embed_dim, with_bias=self.bias, name='q_attn') # Query with the same dimension as the key as specified in `Attention is all you need` paper (section 3.2.2)
        K = hk.Linear(self.k_dim or self.embed_dim, with_bias=self.bias, name='k_attn')
        V = hk.Linear(self.v_dim or self.embed_dim, with_bias=self.bias, name='v_attn')
        # The last proj onto the embed dim
        proj_w = hk.Linear(self.embed_dim, with_bias=self.bias, name='c_proj')
        
        q, k, v = Q(q), K(k), V(v)
        q = rearrange(q, 'B T (h dk) -> B h T dk', h=self.num_heads) 
        k = rearrange(k, 'B T (h dk) -> B h dk T', h=self.num_heads) # Switch the 2 last dim for the matmul with q
        v = rearrange(v, 'B T (h dv) -> B h T dv', h=self.num_heads)

        if self.rotary_encoding:
            raise NotImplementedError("Rotary encoding is not implemented yet.")
        
        key_size = self.embed_dim / self.num_heads
        attn = (q @ k) * (1 / jnp.sqrt(key_size)) 
        if mask is not None:
            attn = jnp.where(mask, attn, float('-inf')) # Fill the non masked values with -inf
        w = jax.nn.softmax(attn, -1) 
        if training:
            w = hk.dropout(hk.next_rng_key(), self.dropout, w)
        y = w @ v # [B, h, T, d]
        y = rearrange(y, 'B h T d -> B T (h d)')
        logits = proj_w(y) # B T C
        if training:
            logits = hk.dropout(hk.next_rng_key(), self.dropout, logits)

        if return_attention:
            return AttrDict(
                    projection=logits, 
                    attention_weights=w
                    )
        return logits


@dataclass
class SelfAttention(hk.Module):
    num_heads : int # The number of heads to divide the embed_dim into
    embed_dim : int # The embedding dimension (d_model in the paper)
    bias : Optional[bool] = False # Whether to use a bias in the linear layers
    dropout : Optional[float] = 0.0 # Dropout regularization
    rotary_encoding : Optional[bool] = False # Whether to use rotary encoding
    name : Optional[str] = None
    
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False, return_attention: bool = True):
        assert self.embed_dim % self.num_heads == 0, "The embed dimension should be divisible by the number of heads."
        # Get a linear layer for the q, k, v at the same time
        qkv_w = hk.Linear(3 * self.embed_dim, with_bias=self.bias, name='c_attn')
        # The last proj onto the embed dim
        proj_w = hk.Linear(self.embed_dim, with_bias=self.bias, name='c_proj')
        
        q, k, v = jnp.split(qkv_w(x), 3, 2) # [B, T, C]
        q = einops.rearrange(q, 'B T (h d) -> B h T d', h=self.num_heads) 
        k = einops.rearrange(k, 'B T (h d) -> B h d T', h=self.num_heads) # Switch the 2 last dim for the matmul with q
        v = einops.rearrange(v, 'B T (h d) -> B h T d', h=self.num_heads)

        if self.rotary_encoding:
            raise NotImplementedError("Rotary encoding is not implemented yet.")
        
        key_size = self.embed_dim / self.num_heads # d_k
        attn = (q @ k) * (1 / jnp.sqrt(key_size)) # [B, h, T, T]
        if mask is not None:
            attn = jnp.where(mask, attn, float('-inf')) # Fill the non masked values with -inf
        w = jax.nn.softmax(attn, -1)  
        if training:
            w = hk.dropout(hk.next_rng_key(), self.dropout, w)
        y = w @ v # [B, h, T, d]
        y = einops.rearrange(y, 'B h T d -> B T (h d)') # Merge the head and the head dim
        logits = proj_w(y) # [B, T, C]
        if training:
            logits = hk.dropout(hk.next_rng_key(), self.dropout, logits) 
        if return_attention:
            return AttrDict(
                    projection=logits, 
                    attention_weights=w, 
                    )
        return logits

class CausalSelfAttention(SelfAttention):
    def __call__(self, x: jnp.ndarray, training: bool = False, return_attention: bool = True):
        _, T, _ = x.shape
        # Build the causal mask (only attention on itself and past values)
        mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))  
        return super().__call__(x, mask=mask, training=training, return_attention=return_attention)
