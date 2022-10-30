#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 25, 14:49:00
@last modified : 2022 October 30, 19:41:34
"""

import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange

from dataclasses import dataclass
from typing import Optional

from relax.utils import AttrDict


@dataclass
class VectorQuantizer(hk.Module):
    embedding_dim: int  # (D,)
    num_embeddings: int  # (K,)
    β: float  # Commitment cost

    def __call__(self, inputs):
        embedding_shape = (self.num_embeddings, self.embedding_dim)  # (K, D)
        embeddings = hk.get_parameter(
            "embeddings",
            embedding_shape,
            init=hk.initializers.VarianceScaling(distribution="uniform"),
        )

        # Reshape the inputs and treat it as a batch of vectors of shape (L, D)
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))

        # Compute the squared L2 norm between the embedding vectors and the flatten input vectors
        # We don't take the exact L2 norm because the square root function is monotonic, the indice of the minimum distance will remain the same
        sq_distances = (
            jnp.sum(jnp.square(flat_inputs), axis=1, keepdims=True).T  # (1,L)
            + jnp.sum(jnp.square(embeddings), axis=1, keepdims=True)  # (K,1)
            - 2 * jnp.matmul(embeddings, flat_inputs.T)  # (K, L)
        )

        # Get the indices of the embedding vector that has the minimal distance regarding each input vectors
        encoding_indices = jnp.argmax(-sq_distances, axis=0)  # (L,)
        one_hot_encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        # Retrieve the closest embedding vector for each input
        encodings = jnp.matmul(one_hot_encodings, embeddings)
        encodings = jnp.reshape(encodings, inputs.shape)
        encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])

        # Loss on the encoding vectors
        # Encoding vectors try to get as close as possible to their corresponding input vectors
        # $\lVert \textmode{sg}[z_{e}(x)] - e\rVert_{2}^{2}$
        codebook_loss = jnp.mean(jnp.square(encodings - jax.lax.stop_gradient(inputs)))

        # Loss on the input vectors
        # Input vectors try to get as close as possible to their corresponding encoding vectors
        # $\lVert z_{e}(x) - \textmode{sg}[e]\rVert_{2}^{2}$
        commitment_loss = jnp.mean(
            jnp.square(jax.lax.stop_gradient(encodings) - inputs)
        )

        # The commitment cost β is their to make sure the encoder commits to an embedding and its output does not grow
        loss = codebook_loss + self.β * commitment_loss

        # Straight through estimator, grad(x) : x
        encodings = inputs + jax.lax.stop_gradient(encodings - inputs)

        return AttrDict(
            encodings=encodings,
            encoding_indices=encoding_indices,
            sq_distances=sq_distances,
            loss=loss,
        )


@dataclass
class VQVAE(hk.Module):
    encoder: hk.Module
    decoder: hk.Module
    embedding_dim: int  # (D,)
    num_embeddings: int  # (K,)
    β: Optional[float] = 0.25

    def vector_quantizer(self, z):
        return VectorQuantizer(self.embedding_dim, self.num_embeddings, self.β)(z)

    def __call__(self, x):
        μ, logvar = self.encoder(x)
        z = self.reparametrize(μ, logvar)
        vq_results = self.vector_quantizer(z)
        logits = self.decoder(vq_results.encodings)

        return AttrDict(logits=logits, μ=μ, logvar=logvar, z=z, vq_results=vq_results)

    @staticmethod
    def reparametrize(μ, logvar):
        eps = jax.random.normal(hk.next_rng_key(), μ.shape)
        σ = jnp.exp(logvar / 2)
        return μ + σ * eps
