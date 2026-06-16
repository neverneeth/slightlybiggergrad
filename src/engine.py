import math
import matplotlib.pyplot as plt
import random
from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)

  build(root)
  return nodes, edges


def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node(name = uid, label = "{ %s }" % (n.label), shape='record')
    if n.op:
      dot.node(name = uid + n.op, label = n.op)
      dot.edge(uid+n.op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)
  return dot


class Tensor:
  def __init__(self, values, _children=(), op='', label='', device='cpu'):
    if device == 'cpu':
      import numpy as xp
    elif device == 'gpu':
      import cupy as xp
    else:
      raise ValueError("Unsupported device")
    self.device = device
    self.xp = xp

    self.data = xp.array(values, dtype=xp.float32)
    self.size = len(values) if isinstance(values, list) else 1
    self.backward = lambda: None
    self._prev = set(_children)
    self.grad = xp.zeros_like(self.data)
    self.op = op
    self.label = label

  def __repr__(self):
    return f"Tensor(Shape: {self.data.shape}, Data: {self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
    assert self.device == other.device, "Tensors must be on the same device"
    out = Tensor(self.data + other.data, (self, other), op='+', device=self.device)
    def _backward():
      grad_self = out.grad
      while len(grad_self.shape) > len(self.data.shape):
        grad_self = grad_self.sum(axis=0)
        # (Optional) If self was shape (1, 5) and out was (32, 5), sum across axis 0 and keep dims
      for i, dim in enumerate(self.data.shape):
        if dim == 1:
          grad_self = grad_self.sum(axis=i, keepdims=True)

      self.grad += grad_self
      grad_other = out.grad
      while len(grad_other.shape) > len(other.data.shape):
        grad_other = grad_other.sum(axis=0)

      for i, dim in enumerate(other.data.shape):
        if dim == 1:
          grad_other = grad_other.sum(axis=i, keepdims=True)

      other.grad += grad_other
    out.backward = _backward
    return out
  
  def __radd__(self, other):
    return self + other
  
  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
    assert self.device == other.device, "Tensors must be on the same device"
    out = Tensor(self.data * other.data, (self, other), op='*', device=self.device)
    def _backward():
      grad_self = out.grad * other.data
      while len(grad_self.shape) > len(self.data.shape):
        grad_self = grad_self.sum(axis=0)
        for i, dim in enumerate(self.data.shape):
          if dim == 1:
            grad_self = grad_self.sum(axis=i, keepdims=True)
      self.grad += grad_self

      grad_other = out.grad * self.data
      while len(grad_other.shape) > len(other.data.shape):
        grad_other = grad_other.sum(axis=0)
      for i, dim in enumerate(other.data.shape):
        if dim == 1:
          grad_other = grad_other.sum(axis=i, keepdims=True)
      other.grad += grad_other
    out.backward = _backward
    return out
  
  def __rmul__(self, other):
    return self * other
  
  def __neg__(self):
    out = Tensor(-self.data, (self,), op='neg', device=self.device)
    def _backward():
      self.grad += -out.grad
    out.backward = _backward
    return out
  
  def __sub__(self, other):
    return self + (-other)
  

  def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
    assert self.device == other.device, "Tensors must be on the same device"
    out = Tensor(self.data @ other.data, (self, other), op='@', device=self.device)
    def _backward():
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad

    out.backward = _backward
    return out
  
  def __radd__(self, other):
    return self + other
  
  def sum(self):
    out = Tensor(self.data.sum(), (self,), op='arrsum', device=self.device)
    def _backward():
      self.grad += self.xp.ones_like(self.data) * out.grad
    out.backward = _backward
    return out
  
  def mean(self):
    out = Tensor(self.data.mean(), (self,), op='arrmean', device=self.device)
    def _backward():
      self.grad += (self.xp.ones_like(self.data) / self.data.size) * out.grad
    out.backward = _backward
    return out

  def tanh(self):
    t = self.xp.tanh(self.data)
    out = Tensor(t, (self,), op='tanh', device=self.device)
    def _backward():
      self.grad += (1- t**2) * out.grad

    out.backward = _backward
    return out
  
  def relu(self):
    t = self.xp.maximum(0, self.data)
    out = Tensor(t, (self,), op='relu', device=self.device)
    def _backward():
      self.grad += (self.data > 0) * out.grad
    out.backward = _backward
    return out
  
  def sigmoid(self):
    clipped_data = self.xp.clip(self.data, -50, 50)
    t =  1 / (1+ self.xp.exp(-clipped_data))
    out  = Tensor(t, (self,), op='sigmoid', device=self.device)
    def _backward():
      self.grad += out.data * (1 - out.data) * out.grad
    out.backward = _backward
    return out
  
  def numpy(self):
    if self.device == 'gpu':
      return self.data.get()
    return self.data

  def backward_all(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = self.xp.ones_like(self.data)
    for node in reversed(topo):
      node.backward()

    
  