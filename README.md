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

**jaxify** lets you apply [JAX](https://github.com/jax-ml/jax) transformations (like [`@jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) and/or [`@jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)) to functions with Python control flow that JAX normally cannot compile, like `if`/`elif`/`else` statements depending on input values.
</div>

## Installation

```bash
pip install jaxify
```

## Getting started

```python
import jax
import jax.numpy as jnp
from jaxify import jaxify

@jax.jit
@jax.vmap
@jaxify
def absolute_value(x):
    if x >= 0:  # <-- If conditional in a JIT-compiled function!
        return x
    else:
        return -x

xs = jnp.arange(-1000, 1000)
ys = absolute_value(xs)  # <-- Runs at JAX speed!
print(ys)
```

## How it works

`@jaxify` is a decorator that transforms Python functions by rewriting their abstract syntax tree (AST) to replace unsupported control flow constructs with JAX-compatible alternatives. It currently supports `if`/`elif`/`else` statements depending on input values, allowing you to write more natural Python code while still benefiting from JAX's performance boost.

When you decorate a function with `@jaxify`, it analyzes the function's source code, identifies control flow constructs, and rewrites them to use JAX's functional control flow primitives (like `jax.lax.cond`). The transformed function is then traceable by JAX, enabling you to apply JAX transformations like `@jax.jit` and `@jax.vmap` seamlessly.

## Compatibility status

The following Python control flow constructs are currently supported within `@jaxify`-decorated functions:


| Python construct        | Support status   | Notes |
|:-----------------------:|:----------------:|:- |
| `if` / `elif` / `else`  | ✅               | Should mostly work |
| `if`-`else` expressions | ⚠️               | Static values only |
| `and` / `or`            | ⚠️               | Static values only. For dynamic values, use [`&` or `jnp.logical_and`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_and.html) / [`\|` or `jnp.logical_or`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_or.html) instead |
| `for` loops             | ❌               | Use [`jax.lax.fori_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html), [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), or [`jax.lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) instead |
| `while` loops           | ❌               | Use [`jax.lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) instead |
| `match`-`case`          | ⚠️               | Static values only. For dynamic values, use an `if`-`elif`-`else` chain or [`jax.lax.switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html) instead |
