#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 October 27, 14:52:06
@last modified : 2022 October 27, 14:59:47
"""

from setuptools import setup, find_packages

def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()

setup(
    name='relax',
    version='0.0.0a',
    url='https://github.com/romaingrx/relax',
    author='Romain Graux',
    description='Implementation of ML papers in JAX+Haiku+Optax while being relaxed',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['relax*']),
    author_email='romaingrx@duck.com',
    install_requires=_parse_requirements('requirements.txt'),
    zip_safe=False,
)
