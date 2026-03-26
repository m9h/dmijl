# Shared numerical utilities

"""Softplus activation: log(1 + exp(x)), numerically stable."""
softplus(x) = x > 20 ? x : log(1 + exp(x))

"""Sigmoid activation: 1 / (1 + exp(-x))."""
_sigmoid(x) = 1 / (1 + exp(-x))
