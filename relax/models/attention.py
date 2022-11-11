#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 November 11, 17:43:12
@last modified : 2022 November 11, 18:51:52
"""

import jax
import jax.numpy as jnp
import haiku as hk
import einops

from typing import Optional

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


    def __call__(self, query, key, value, mask = None, return_attention_weights: bool = False):
        """
            queries (batch_size, sequence_length, embed_dim)
            keys (batch_size, sequence_length, embed_dim)
            values (batch_size, sequence_length, embed_dim)
        """
        separate_heads_fct = lambda x: einops.rearrange(x, "b seq_length (num_heads key_size) -> b num_heads seq_length key_size", num_heads=self.num_heads)

        query_heads = separate_heads_fct(hk.Linear(self.embed_dim, w_init=self.w_init, name="Q")(query)) # [batch, num_heads, sequence_length, key_size]
        key_heads = separate_heads_fct(hk.Linear(self.embed_dim, w_init=self.w_init, name="K")(key)) # [batch, num_heads, sequence_length, key_size]        
        value_heads = separate_heads_fct(hk.Linear(self.embed_dim, w_init=self.w_init, name="V")(value)) # [batch, num_heads, sequence_length, key_size]       

        attention_logits = jnp.einsum("bhsd,bhSd->bhsS", query_heads, key_heads) # Multiply each query heads by each key heads to get the attention map
        attention_logits /= jnp.sqrt(self.key_size) # Scale it with the square root of the key size

        if mask is not None:
            attention_logits = jnp.where(mask, attention_logits, 1e-31)

        attention_weights = jax.nn.softmax(attention_logits)

        attention = jnp.einsum("bhsS,bhSd->bshd", attention_weights, value_heads)
        attention = einops.rearrange(attention, "b s h d -> b s (h d)")

        projection = hk.Linear(self.embed_dim, w_init=self.w_init, name="projection")(attention)

        if return_attention_weights:
            return projection, attention_weights
        return projection
