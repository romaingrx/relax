#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 22, 17:40:37
@last modified : 2022 October 22, 18:06:27
"""

import haiku as hk
from typing import Union, Callable


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def filter_params(f: Union[str, Callable], params: hk.Params) -> hk.Params:
    cond = f if callable(f) else lambda x, y: f in x
    return {k: v for k, v in params.items() if cond(k, v)}
