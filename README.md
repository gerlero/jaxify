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
|:----------------------------------------------------------:|
| You're welcome to try it out and report any issues!        |

---

**jaxify** lets you apply [JAX](https://github.com/jax-ml/jax) transformations (like [`@jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) and/or [`@jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)) to functions with common Python constructs that JAX cannot itself handle, such as `if` conditions that depend on input values.
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
@jaxify  # <-- Just decorate your function with @jaxify
def absolute_value(x):
    if x >= 0:  # <-- If block in a JIT-compiled function
        return x
    else:
        return -x

xs = jnp.arange(-1000, 1000)
ys = absolute_value(xs)  # <-- Runs at JAX speed!
print(ys)
```

## How it works

The `@jaxify` decorator transforms Python functions using a mixture of static analysis and dynamic tracing to replace unsupported Python constructs with JAX-compatible alternatives. After the transformations, the functions become traceable by JAX, enabling you to apply functional JAX transformations like `@jax.jit` and `@jax.vmap` in a seamless manner.

## Compatibility status

The following Python constructs are currently supported within `@jaxify`-decorated functions:

### 🔀 Conditionals

| Construct                               | Works? | Notes |
|:---------------------------------------:|:------:|:------|
| `if` statements                         | ✅     | Fully supported including `elif` and `else` clauses. All branches are traced and translated to calls to [`jax.lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) |
| `if` expressions (e.g. `a if b else c`) | ✅     | Traced and translated to [`jax.lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) |

### ⚖️ Comparisons

| Construct                        | Works? | Notes |
|:--------------------------------:|:------:|:------|
| `==`, `!=`, `<`, `>`, `<=`, `>=` | ✅     | Chained comparisons (e.g. `x < y <= z`) are supported by translation to the equivalent chain of individual comparisons |

### 1️⃣ Logical operators

| Construct    | Works? | Notes |
|:------------:|:------:|:------|
| `and` / `or` | ✅     | Short-circuiting of traced values supported via translation to [`jax.lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) calls |
| `not`        | ✅     | Translates to [`jnp.logical_not`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_not.html) for traced single values |

### 🔄 Loops

| Construct     | Works? | Notes |
|:-------------:|:------:|:------|
| `for` loops   | ❌     | Currently unsupported. Use [`jax.lax.fori_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html), [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), or [`jax.lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) instead |
| `while` loops | ❌     | Currently unsupported. Use [`jax.lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) instead |


### 🎯 Pattern matching

| Construct      | Works? | Notes |
|:--------------:|:------:|:------|
| `match`-`case` | ✅⚠️   | Static values only. For traced values, use an `if`-`elif`-`else` chain or [`jax.lax.switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html) instead |
