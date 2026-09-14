"""
Legacy setup.py shim for compatibility with tools that don't yet read
pyproject.toml directly (e.g. `pip install -e .` on older pip versions).
All actual package metadata lives in pyproject.toml.
"""
from setuptools import setup

setup()