<div align="center">

# jaxify

Write **Python**. Run **JAX**.

[![CI](https://github.com/gerlero/jaxify/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/jaxify/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/jaxify/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/jaxify)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/jaxify/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/jaxify/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/jaxify)](https://pypi.org/project/jaxify/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxify)](https://pypi.org/project/jaxify/)

| ⚠️ **jaxify** is an experimental project under development |
|:---:|
| Feel free to test out and report any issues. _Do not use in production_. |

---

**jaxify** lets you JIT-compile functions (using [JAX](https://github.com/jax-ml/jax)) that [`@jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) cannot handle. With **jaxify**, you can compile functions with e.g. Python `if`/`elif`/`else` statements (with support for other control flow structures planned for the future) that might be affected by the values of inputs.

**jaxify**'s `@jitx` decorator works exclusively on the decorated function and intervenes only at tracing/compilation time; it does not have any effect at actual runtime besides the code it emits for JAX.

</div>

## Installation

```bash
pip install jaxify
```

## Getting started

```python
import jax
import jax.numpy as jnp
from jaxify import jitx

@jax.vmap
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
