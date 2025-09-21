# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Submodule defining the environment definitions."""

from .vec_env import VecEnv
from .safe_vec_env import SafeVecEnv

__all__ = ["VecEnv", "SafeVecEnv"]
