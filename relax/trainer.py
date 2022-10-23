#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 21, 16:25:35
@last modified : 2022 October 22, 17:40:14
"""

import jax
import optax
import haiku as hk
import jax.numpy as jnp

from functools import partial
from dataclasses import dataclass
from typing import Union, Callable, Sequence, NamedTuple

from .metrics import MeanMetric
from tqdm.auto import tqdm, trange

class Batch(jnp.ndarray):
    def __hash__(self):
        return hash(id(self))
    def __eq__(self, other):
        return hash(self) == hash(other)

Dataset = Sequence[Batch]

@dataclass
class TrainingConfig:
    epochs: int = 1
    number_of_steps: Union[None, int] = None
    evaluation_frequency: Union[None, int] = None

    def __hash__(self):
        return hash(str(self.epochs) + str(self.number_of_steps) + str(self.evaluation_frequency))

    def __eq__(self, other):
        return hash(self) == hash(other)

class TrainingState(NamedTuple):
    rng: jax.random.PRNGKey
    params: hk.Params
    opt_state: optax.OptState

    def __hash__(self):
        return hash(str(self.rng) + str(self.params) + str(self.opt_state))

    def __eq__(self, other):
        return hash(self) == hash(other)

@dataclass
class Trainer:
    model: hk.Transformed
    optimizer: optax.GradientTransformation
    config: TrainingConfig = None

    def init(
            self,
            rng: Union[jax.random.PRNGKey, int],
            *args,
            **kwargs
            ) -> TrainingState:
        model_params = self.model.init(rng, *args, **kwargs)
        optimizer_state = self.optimizer.init(model_params)
        state = TrainingState(
                rng,
                model_params,
                optimizer_state
                )
        self.config = self.config or TrainingConfig()
        return state

    def apply(self, *args, **kwargs):
        return self.model.apply(*args, **kwargs)

    def train(self, state: TrainingState, loss_fn: Callable, ds : Dataset, jit_update_step:bool=True, jit_epoch_loop:bool=False) -> TrainingState:

        def __update(state : TrainingState, loss_fn: Callable, batch : Batch) -> Union[TrainingState, float]:
            rng, next_rng = jax.random.split(state.rng)
            loss, grads = jax.value_and_grad(loss_fn)(state.params, next_rng, batch)
            updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            new_state = TrainingState(next_rng, new_params, new_opt_state)
            return new_state, loss
        update = jax.jit(__update, static_argnums=(1)) if jit_update_step else __update

        def __update_epoch(state : TrainingState, config : TrainingConfig, loss_fn: Callable, ds : Dataset) -> TrainingState:
            loss_metric = MeanMetric("loss")
            for step, batch in enumerate(ds):
                state, loss = update(state, loss_fn, batch)
                loss_metric.update(loss)

                if config.number_of_steps and config.number_of_steps <= step:
                    break

            return state, loss_metric.result()
        update_epoch = jax.jit(__update_epoch, static_argnums=(1, 2)) if jit_epoch_loop else __update_epoch

        bar = trange(self.config.epochs, desc=f"Training", unit="epoch")
        for epoch in bar:
            state, loss = update_epoch(state, self.config, loss_fn, ds)
            bar.set_postfix({"loss": f"{loss:.3e}"})
        return state
