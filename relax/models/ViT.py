#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 November 19, 11:49:18
@last modified : 2022 November 21, 11:21:36
"""

import jax
import einops
import haiku as hk
from functools import partial
from dataclasses import dataclass
from relax.models import MultiHeadAttention

from typing import Union, Sequence, Tuple, Optional

def extract_patches(img, patch_shape=(8, 8), padding="VALID"):
    h, w, c = img.shape
    ph, pw = patch_shape
    if padding == "VALID":
        img = img[:h // ph * ph, :w // pw * pw]
    elif padding == "SAME":
        img = jax.numpy.zeros((
            h + h % ph,
            w + w % pw,
            c
            )).at[:h, :w].set(img)
    else:
        raise Exception(f"Padding method `{padding}` not known")
    return einops.rearrange(img, "(ph p1) (pw p2) c -> (p1 p2) ph pw c", ph=ph, pw=pw)

@dataclass
class PatchesEncoder(hk.Module):
    patch_shape: Union[int, Tuple[int, int]]
    patch_embedding_dim: int
    padding: Optional[str] = "VALID"
    
    def project_patches(self, patches):
        return hk.Linear(self.patch_embedding_dim)(patches)

    @partial(jax.vmap, in_axes=(None, 0))
    def embed_positions(self, patches):
        n_patches = patches.shape[0]
        positions = jax.numpy.arange(n_patches)
        return hk.Embed(vocab_size=n_patches, embed_dim=self.patch_embedding_dim)(positions)

    def __call__(self, x):
        # Ensure that the patch shape is a tuple
        self.patch_shape = (self.patch_shape, self.patch_shape) if isinstance(self.patch_shape, int) else self.patch_shape
        # Extract patches for each batch element
        patches = jax.vmap(extract_patches, (0, None, None))(x, self.patch_shape, self.padding)
        # Flatten each patch
        flattened_patches = einops.rearrange(patches, "b p ... -> b p (...)")
        # Project patches to the embedding dimension
        projections = self.project_patches(flattened_patches)
        # Embed positions 
        position_embeddings = self.embed_positions(patches)
        # Add position embeddings to the projections
        # TODO: Add a learnable scaling factor
        return projections + position_embeddings

@dataclass
class ViT(hk.Module):
    patch_shape: Union[int, Tuple[int, int]]
    patch_embedding_dim: int
    padding: Optional[str] = "VALID"
    n_heads: Optional[int] = 8
    n_transformer_blocks: Optional[int] = 1
    mlp_dims: Optional[Sequence[int]] = (256, 128)
    mlp_dropout: Optional[float] = 0.1


    def __call__(self, x):

        # TODO: check if haiku has an easy way to rename a functionnal module, otherwise implement a simple decorator that turns a func to a renamed module
        @dataclass
        class MLP(hk.Module):
            dropout_rate: float
            units: Sequence[int]
            name: Optional[str] = "MLP"

            def __call__(self, x):
                for units in self.units:
                    x = hk.Linear(units)(x)
                    x = jax.nn.gelu(x)
                    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
                return x

        @dataclass
        class TransformerBlock(hk.Module):
            n_heads: int
            mlp_dims: Sequence[int]
            mlp_dropout: float
            embedding_dim: int
            name: Optional[str] = "TransformerBlock"

            def __call__(self, x):
                z_ = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x)
                z_ = MultiHeadAttention(self.n_heads, self.embedding_dim)(z_, z_, z_).projection
                x_ = x + z_ # Residual connection
                z_ = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x_)
                z_ = MLP(self.mlp_dropout, self.mlp_dims + (self.embedding_dim,))(z_)
                return x_ + z_ # Residual connection

        patches_encoder = PatchesEncoder(self.patch_shape, self.patch_embedding_dim, self.padding)
        x = patches_encoder(x)
        for _ in range(self.n_transformer_blocks):
            x = TransformerBlock(self.n_heads, self.mlp_dims, self.mlp_dropout, self.patch_embedding_dim)(x)
        return x
