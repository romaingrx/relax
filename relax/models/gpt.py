#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 February 03, 18:14:12
@last modified : 2023 March 12, 23:13:05
"""

import jax
import math
import einops
import haiku as hk
import jax.numpy as jnp
from relax import nn
from relax.models import CausalSelfAttention

from typing import Optional
from functools import partial
from dataclasses import dataclass

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


@dataclass
class MLP(hk.Module):
    n_embed : int
    dropout: float
    bias: Optional[bool] = False
    name: Optional[str] = "MLP"

    def __call__(self, x, training: bool = True):
        x = hk.Linear(4 * self.n_embed, with_bias=self.bias, name='c_fc')(x)
        x = new_gelu(x)
        x = hk.Linear(self.n_embed, with_bias=self.bias, name='c_proj')(x)
        if training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        return x

@dataclass
class GPTBlock(hk.Module):
    n_embed : int
    n_head : int
    dropout : float 
    bias : Optional[bool] = False
    name : str = "GPTBlock" 
    
    def __call__(self, x, training: bool = True):
        x = nn.LayerNorm(self.n_embed, beta=self.bias, name='ln_1')(x) # Normalize the batch
        x = CausalSelfAttention(self.n_head, self.n_embed, bias=self.bias, dropout=self.dropout, name="attn")(x).projection # Apply the self causal attention
        x += x # Skip connection
        x = nn.LayerNorm(self.n_embed, beta=self.bias, name='ln_2')(x) # Normalize the skipped connection
        x = MLP(self.n_embed, self.dropout, self.bias, name='mlp')(x, training) # Apply the MLP
        x += x 
        return x

@dataclass
class GPT(hk.Module):
    vocab_size : int
    block_size : int
    n_blocks : int
    n_embed : int
    n_head : int 
    dropout : int
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
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        for i in range(self.n_blocks):
            x = GPTBlock(self.n_embed, self.n_head, self.dropout, self.bias, name=f"block_{i}")(x, training)
        x = nn.LayerNorm(self.n_embed, beta=self.bias, name='ln_f')(x) 
        # There is a possibility to speed-up the inference time by just applying the lm_head on the last position
        logits = hk.Linear(self.vocab_size, with_bias=False, name="lm_head")(x) 
        return logits

    @classmethod
    def from_pretrained(self, model_type:str, dropout:float=0):
        import re
        from relax.utils import treedef_flatten, treedef_unflatten
        from transformers import FlaxGPT2LMHeadModel, AutoTokenizer

        hf_gpt = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        hf_params = dict(treedef_flatten(hf_gpt.params))

        # Adapt the parameters names to the ones used in the relax library
        rules = (
            lambda x: re.sub('/bias$', '/offset', x) if re.search(r"/ln_([\d+f])/bias", x) else x,
            lambda x: re.sub('/bias$', '/b', x),
            lambda x: re.sub('/h/', '/block_', x),
            lambda x: re.sub('/kernel$', '/w', x),
            lambda x: re.sub('/embedding$', '/embeddings', x),
        )
        
        # Transpose the parameters that are not in the same order
        transposed = [
                'attn/c_attn/w', 'attn/c_proj/w', 'mlp/c_fc/w', 'mlp/c_proj/w' 
                ]
        
        params = {}
        for k, v in hf_params.items():
            for rule in rules:
                k = rule(k)
            params[k] = v.T if any(k.endswith(t) for t in transposed) else v
            
        # All GPT2 models share their lm_head weights with the token embeddings (wte) weights but obviously transposed to project the embeddings to the vocab size
        params['transformer/lm_head/w'] = params['transformer/wte/embeddings'].T
        params = treedef_unflatten(params)

        common_args = dict(
            name='transformer', # match hf naming
            vocab_size=50257, # always the same for all gpt2 models
            block_size=1024, # always the same for all gpt2 models
            bias=True, # always the same for all gpt2 models
            dropout=dropout, # override default
        )
            
        config_args = {
                'gpt2':         dict(n_blocks=12, n_head=12, n_embed=768),  # 124M params
                'gpt2-medium':  dict(n_blocks=24, n_head=16, n_embed=1024), # 350M params
                'gpt2-large':   dict(n_blocks=36, n_head=20, n_embed=1280), # 774M params
                'gpt2-xl':      dict(n_blocks=48, n_head=25, n_embed=1600), # 1558M params
            }[model_type]

        return {**common_args, **config_args}, params
