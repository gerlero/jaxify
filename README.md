<div align="center">

# jaxify

**Write Python. Run JAX.**

[![CI](https://github.com/gerlero/jaxify/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/jaxify/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/jaxify/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/jaxify)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/jaxify/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/jaxify/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/jaxify)](https://pypi.org/project/jaxify/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxify)](https://pypi.org/project/jaxify/)

---

| ⚠️ **jaxify** is an experimental project under development |
|:---:|
| Right now, only some `if` statements may work. Use at your own risk. |

</div>

## Installation

```bash
pip install jaxify
```

## Getting started

```python
import jax.numpy as jnp
from jaxify import jitx

@jitx
def absolute_value(x):
    if x >= 0:  # <-- If conditional in a JIT-compiled function!
        return x
    else:
        return -x

xs = jnp.arange(-1000, 1000)
ys = absolute_value(xs)  # <-- Runs at JAX speed!
print(ys)
```
