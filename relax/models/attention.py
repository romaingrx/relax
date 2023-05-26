#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 November 11, 17:43:12
@last modified : 2023 May 26, 15:15:21
"""

import jax
import jax.numpy as jnp
import haiku as hk
import einops

from relax.utils import AttrDict

from typing import Optional
from functools import cached_property
from dataclasses import dataclass

# TODO: create a faster operation for dot product attention (see deepspeed or the latest pytorch 2.0 implementation)

@dataclass
class RotaryEncoding():
    """Rotary Encoding for attention."""
    
    dim : int
    θ : float = 10_000

    @cached_property
    def inv_freq(self):
        return 1. / (self.θ ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))

    def pos_enc(self, t):
        pos_enc = jnp.einsum('t, f -> tf', jnp.arange(t), self.inv_freq)
        return einops.repeat(pos_enc, '... f -> ... (f r)', r=2)

    @staticmethod
    def rotate_half(u):
        # last dimension of u is [u1, u2, u3, u4, ...]
        u1, u2 = einops.rearrange(u, '... (d r) -> r ... d', r=2)
        u = jnp.stack((-u2, u1), axis=-1)
        # last dimension of result is [-u2, u1, -u4, u3, ...]
        return einops.rearrange(u, '... d r -> ... (d r)')

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Rotary encoding for attention.

        Args:
            u (jnp.ndarray): either queries or keys of shape [batch_size, heads, seq_len, dim]

        Returns:
            jnp.ndarray: the rotated queries or keys of shape [batch_size, heads, seq_len, dim]
        """
        num_tokens = list(u.shape)[-2]
        pos_enc = self.pos_enc(num_tokens)[:num_tokens]
        # i % 2 == 0 : 
        #      rotatedᵢ   = uᵢ cosθᵢ - uᵢ+1 sinθᵢ
        #      rotatedᵢ+1 = uᵢ+1 cosθᵢ + uᵢ sinθᵢ
        return u * jnp.cos(pos_enc) + self.rotate_half(u) * jnp.sin(pos_enc)

    def rotate_qk(self, q, k):
        # q and k are [B, H, T, D]
        return self(q), self(k)

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

    @cached_property
    def rotary_encoder(self):
        return RotaryEncoding(self.embed_dim // self.num_heads)
    
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
        k = rearrange(k, 'B T (h dk) -> B h T dk', h=self.num_heads) 
        v = rearrange(v, 'B T (h dv) -> B h T dv', h=self.num_heads)

        if self.rotary_encoding:
            q, k = self.rotary_encoder.rotate_qk(q, k)
        
        k = einops.rearrange(k, 'B h T d -> B h d T') # Switch the 2 last dim for the matmul with q
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

    @cached_property
    def rotary_encoder(self):
        return RotaryEncoding(self.embed_dim // self.num_heads)
    
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False, return_attention: bool = True):
        assert self.embed_dim % self.num_heads == 0, "The embed dimension should be divisible by the number of heads."
        # Get a linear layer for the q, k, v at the same time
        qkv_w = hk.Linear(3 * self.embed_dim, with_bias=self.bias, name='c_attn')
        # The last proj onto the embed dim
        proj_w = hk.Linear(self.embed_dim, with_bias=self.bias, name='c_proj')
        
        q, k, v = jnp.split(qkv_w(x), 3, 2) # [B, T, C]
        q = einops.rearrange(q, 'B T (h d) -> B h T d', h=self.num_heads) 
        k = einops.rearrange(k, 'B T (h d) -> B h T d', h=self.num_heads)
        v = einops.rearrange(v, 'B T (h d) -> B h T d', h=self.num_heads)

        if self.rotary_encoding:
            q, k = self.rotary_encoder.rotate_qk(q, k)
        
        k = einops.rearrange(k, 'B h T d -> B h d T') # Switch the 2 last dim for the matmul with q
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
