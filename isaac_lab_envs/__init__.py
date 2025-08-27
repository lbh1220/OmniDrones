# MIT License
#
# Copyright (c) 2023 Isaac Lab Forest Environment Implementation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Isaac Lab Forest Environment Implementation

This package contains implementations of the Forest navigation environment
using both Direct RL and Manager-based workflows compatible with Isaac Lab.
"""

__version__ = "1.0.0"

from .direct.forest_env import ForestEnv, ForestEnvCfg
from .manager_based.forest_env_cfg import ForestManagerEnvCfg

__all__ = [
    "ForestEnv",
    "ForestEnvCfg", 
    "ForestManagerEnvCfg",
] 