#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 21, 19:04:35
@last modified : 2022 October 21, 19:52:49
"""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    state: list = None

    def update(self, value):
        if self.state:
            self.state.append(value)
        else:
            self.state = [value]

    def result(self):
        return self.state

    def reset(self):
        self.state = None


class MeanMetric(Metric):
    def result(self):
        state = self.state
        return jnp.mean(jnp.array(state))
