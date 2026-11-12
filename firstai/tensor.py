"""Small, dependency-free 2D tensor engine with reverse-mode autodiff.

This is educational infrastructure for FirstAI. It deliberately uses Python
lists so every operation and gradient remains inspectable.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        if not isinstance(data, list) or not data:
            raise ValueError("Tensor data must be a non-empty 2D list")
        if not isinstance(data[0], list) or not data[0]:
            raise ValueError("Tensor data must be a non-empty 2D list")
        width = len(data[0])
        if any(not isinstance(row, list) or len(row) != width for row in data):
            raise ValueError("Tensor must be rectangular")
        self.data = [[float(v) for v in row] for row in data]
        self.rows, self.cols = len(data), width
        self.grad = [[0.0] * width for _ in range(len(data))]
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self):
        return (self.rows, self.cols)

    def _check_same(self, other):
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor([[other] * self.cols for _ in range(self.rows)])
        self._check_same(other)
        out = Tensor([[self.data[i][j] + other.data[i][j]
                       for j in range(self.cols)] for i in range(self.rows)],
                     (self, other), "add")
        def backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += out.grad[i][j]
                    other.grad[i][j] += out.grad[i][j]
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor([[-v for v in row] for row in self.data], (self,), "neg")
        def backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] -= out.grad[i][j]
        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor([[v * float(other) for v in row] for row in self.data],
                         (self,), "scale")
            def backward():
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += float(other) * out.grad[i][j]
            out._backward = backward
            return out
        self._check_same(other)
        out = Tensor([[self.data[i][j] * other.data[i][j]
                       for j in range(self.cols)] for i in range(self.rows)],
                     (self, other), "mul")
        def backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += other.data[i][j] * out.grad[i][j]
                    other.grad[i][j] += self.data[i][j] * out.grad[i][j]
        out._backward = backward
        return out

    __rmul__ = __mul__

    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply shapes {self.shape} and {other.shape}")
        out = Tensor([[sum(self.data[i][k] * other.data[k][j]
                           for k in range(self.cols))
                       for j in range(other.cols)] for i in range(self.rows)],
                     (self, other), "matmul")
        def backward():
            for i in range(self.rows):
                for k in range(self.cols):
                    for j in range(other.cols):
                        self.grad[i][k] += out.grad[i][j] * other.data[k][j]
                        other.grad[k][j] += self.data[i][k] * out.grad[i][j]
        out._backward = backward
        return out

    def transpose(self):
        out = Tensor([[self.data[i][j] for i in range(self.rows)]
                      for j in range(self.cols)], (self,), "transpose")
        def backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += out.grad[j][i]
        out._backward = backward
        return out

    def map(self, fn, derivative, name):
        out = Tensor([[fn(v) for v in row] for row in self.data],
                     (self,), name)
        def backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += derivative(self.data[i][j]) * out.grad[i][j]
        out._backward = backward
        return out

    def tanh(self):
        return self.map(math.tanh, lambda x: 1.0 - math.tanh(x) ** 2, "tanh")

    def sigmoid(self):
        def sig(x):
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)
        return self.map(sig, lambda x: sig(x) * (1.0 - sig(x)), "sigmoid")

    def relu(self):
        return self.map(lambda x: max(0.0, x), lambda x: 1.0 if x > 0 else 0.0, "relu")

    def sum(self):
        out = Tensor([[sum(sum(row) for row in self.data)]], (self,), "sum")
        def backward():
            amount = out.grad[0][0]
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += amount
        out._backward = backward
        return out

    def mean(self):
        return self.sum() * (1.0 / (self.rows * self.cols))

    def zero_grad(self):
        for row in self.grad:
            for j in range(len(row)):
                row[j] = 0.0

    def backward(self):
        if self.shape != (1, 1):
            raise ValueError("backward() requires a scalar tensor; call sum() or mean()")
        topo, visited = [], set()
        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)
        build(self)
        self.grad[0][0] = 1.0
        for node in reversed(topo):
            node._backward()

    def tolist(self):
        return [row[:] for row in self.data]

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"


def layer_norm(x: Tensor, eps=1e-5):
    """Normalize each row independently (feature dimension is columns)."""
    out_rows = []
    means, inv_stds = [], []
    for row in x.data:
        mean = sum(row) / len(row)
        variance = sum((v - mean) ** 2 for v in row) / len(row)
        inv = 1.0 / math.sqrt(variance + eps)
        means.append(mean)
        inv_stds.append(inv)
        out_rows.append([(v - mean) * inv for v in row])
    out = Tensor(out_rows, (x,), "layer_norm")
    def backward():
        n = x.cols
        for i, row in enumerate(x.data):
            g = out.grad[i]
            centered = [v - means[i] for v in row]
            inv = inv_stds[i]
            sum_g = sum(g)
            sum_g_centered = sum(g[j] * centered[j] for j in range(n))
            for j in range(n):
                x.grad[i][j] += (inv / n) * (
                    n * g[j] - sum_g - centered[j] * (inv ** 2) * sum_g_centered
                )
    out._backward = backward
    return out


def softmax_rows(x: Tensor):
    """Numerically stable softmax across each row."""
    rows = []
    probs = []
    for row in x.data:
        peak = max(row)
        exps = [math.exp(v - peak) for v in row]
        total = sum(exps)
        p = [v / total for v in exps]
        rows.append(p)
        probs.append(p)
    out = Tensor(rows, (x,), "softmax")
    def backward():
        for i, p in enumerate(probs):
            dot = sum(out.grad[i][j] * p[j] for j in range(x.cols))
            for j in range(x.cols):
                x.grad[i][j] += p[j] * (out.grad[i][j] - dot)
    out._backward = backward
    return out
