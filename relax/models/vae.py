#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 14, 10:08:57
@last modified : 2022 October 23, 23:00:30
"""

import jax
import jax.numpy as jnp
from jax import random, nn
import haiku as hk

from typing import Optional
from dataclasses import dataclass

@dataclass
class VAE(hk.Module):
    encoder: hk.Module
    decoder: hk.Module

    def __call__(self, x):
        μ, logvar = self.encoder(x)
        z = VAE.reparametrize(μ, logvar)
        logits = self.decoder(z)

        return logits, μ, logvar

    @staticmethod
    def reparametrize(μ, logvar):
        eps = random.normal(hk.next_rng_key(), μ.shape)
        σ = jnp.exp(logvar / 2)
        return μ + σ * eps

    @staticmethod
    def loss_fn(x_hat, mean, logvar, batch):

        @jax.vmap
        def kl_divergence(mean, logvar):
            return -.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

        @jax.vmap
        def mse(logits, labels):
            return jnp.sum(jnp.square(logits - labels))


        return kl_divergence(mean, logvar) + mse(x_hat, batch)

