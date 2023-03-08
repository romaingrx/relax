#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 22, 17:40:37
@last modified : 2023 March 08, 17:26:39
"""

import haiku as hk
from typing import Union, Callable

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def tree_filter(f: Union[str, Callable], params: hk.Params) -> hk.Params:
    cond = f if callable(f) else lambda x, y: f in x
    return {k: v for k, v in params.items() if cond(k, v)}

def treedef_flatten(tree, sep="/"):
    def _treedef_flatten(tree, prefix):
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from _treedef_flatten(v, k if prefix is None else f"{prefix}{sep}{k}")
        elif prefix is not None:
            yield prefix, tree
        else:
            yield tree
    return _treedef_flatten(tree, None)

def treedef_unflatten(tree, sep="/"):
    unflattened_tree = {}
    for k, v in tree.items():
        if sep in k:
            prefix, suffix = k.rsplit(sep, 1)
            if prefix not in unflattened_tree:
                unflattened_tree[prefix] = {}
            unflattened_tree[prefix].update({suffix:v})
        else:
            unflattened_tree[k] = v
    return unflattened_tree




