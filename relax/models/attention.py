#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 November 11, 17:43:12
@last modified : 2023 March 08, 17:02:25
"""

import jax
import jax.numpy as jnp
import haiku as hk
import einops

from relax.utils import AttrDict

from typing import Optional
from dataclasses import dataclass

class MultiHeadAttention(hk.Module):
    def __init__(self, 
            num_heads, 
            embed_dim, 
            w_init: Optional[hk.initializers.Initializer] = None, 
            name: Optional[str] = None
            ):
        super().__init__(name=name)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.num_heads == 0, "The embed dimension should be divisible by the number of heads."
        self.key_size = self.embed_dim // self.num_heads

        self.w_init = w_init 


    def __call__(self, query, key, value, mask = None):
        """
            queries (batch_size, sequence_length, embed_dim)
            keys (batch_size, sequence_length, embed_dim)
            values (batch_size, sequence_length, embed_dim)
        """
        q, k, v = jnp.split(hk.Linear(3 * self.embed_dim, w_init=self.w_init, name="c_attn")(query), 3, axis=-1)

        query_heads = einops.rearrange(q, "b s (h d) -> b h s d", h=self.num_heads) # [batch, num_heads, sequence_length, key_size]
        key_heads = einops.rearrange(q, "b s (h d) -> b h s d", h=self.num_heads) # [batch, num_heads, sequence_length, key_size]        
        value_heads = einops.rearrange(v, "b s (h d) -> b h s d", h=self.num_heads) # [batch, num_heads, sequence_length, key_size]        

        attention_logits = jnp.einsum("bhsd,bhSd->bhsS", query_heads, key_heads) # Multiply each query heads by each key heads to get the attention map
        attention_logits /= jnp.sqrt(self.key_size) # Scale it with the square root of the key size
        if mask is not None:
            # Set the False values to -∞ so that the softmax values will be set to 0
            attention_logits = jnp.where(mask, attention_logits, jnp.finfo(attention_logits.dtype).min)

        attention_weights = jax.nn.softmax(attention_logits)

        attention = jnp.einsum("bhsS,bhSd->bshd", attention_weights, value_heads)
        attention = einops.rearrange(attention, "b s h d -> b s (h d)")

        projection = hk.Linear(self.embed_dim, w_init=self.w_init, name="c_proj")(attention)

        return AttrDict(
                projection=projection, 
                attention_weights=attention_weights, 
                )

class SelfAttention(MultiHeadAttention):
    def __init__(self, 
            num_heads, 
            embed_dim, 
            w_init: Optional[hk.initializers.Initializer] = None, 
            name: Optional[str] = None
            ):
        super().__init__(num_heads, embed_dim, w_init, name)

    def __call__(self, x, mask = None):
        return super().__call__(x, x, x, mask)

# TODO: speedup masked matmul 
class CausalSelfAttention(SelfAttention):
    def __init__(self, 
            num_heads, 
            embed_dim, 
            w_init: Optional[hk.initializers.Initializer] = None, 
            name: Optional[str] = None
            ):
        super().__init__(num_heads, embed_dim, w_init, name)

    def __call__(self, x):
        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=x.dtype), k=0)
        return super().__call__(x, mask)
