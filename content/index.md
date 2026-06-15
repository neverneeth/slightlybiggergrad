---
title: "Engineering Log: Building slightlybiggergrad"
date: 2026-06-15
draft: false
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "A deep dive into constructing an accelerated, hardware-agnostic autograd tensor engine from scratch."
disableShare: false
---

### Building `slightlybiggergrad`: An Accelerated, Device-Agnostic Autograd Engine from Scratch

This is my attempt at creating a tutorial for **`slightlybiggergrad`**. This library is an N-dimensional matrix autograd engine built from scratch in pure Python. It is designed to be an educational bridge between scalar-based autodiff engines (like Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)) and modern, hardware-accelerated frameworks (like PyTorch and JAX).

I highly recommend watching the [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) before diving into this tutorial, as it covers the core concepts of automatic differentiation in a simple scalar context. `slightlybiggergrad` takes those concepts and extends them to operate directly on N-dimensional tensors, while also incorporating device-agnostic hardware acceleration using CuPY.

#### Contents

1. [Introduction](neverneeth.github.io/slightlybiggergrad/tut/introduction.html/)
