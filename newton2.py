# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

def deriv(f, x, eps = 1e-5):
    return (f(x+eps) - f(x)) / eps


def deriv2(f, x, eps = 1e-5):
    return (deriv(f, x+eps, eps) - deriv(f, x, eps)) / eps

def optimize(x0, f, tol =1e-4):
    x_new = x0 - deriv(f, x0)/ deriv2(f, x0)
    x = x0
    while abs(x_new - x) > tol:
        x = x_new
        x_new = x0 - deriv(f, x) / deriv2(f, x)
    return {"x": x_new,
            'value': f(x_new)}

def multivariate(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        g = grad_f(x) 
        H = hess_f(x)  
        if np.linalg.solve(H,-g) < tol:
            return x
        try:
            delta_x = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian inconvertable, failed to solve the minimum")

        x += delta_x

    return x

# %%
